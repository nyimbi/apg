# World-Class Improvements — Clinical Management (healthcare_cli)

**Capability**: `healthcare_cli` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Early Warning Score (EWS) Engine

**Problem**: Sepsis screening is isolated. No generalised deterioration tracking across vital sign streams.

**Improvement**: Implement a continuous EWS engine that aggregates NEWS2/MEWS/PEWS scores from incoming vital sign events, persists them per admission, and auto-fires a CDS alert when a threshold is breached. Each score record carries the sub-score components so clinicians can see which parameter triggered escalation.

**Impact**: Reduces time-to-recognition of deteriorating patients by providing a single auditable trigger chain from raw vitals to escalation alert.

---

## 2. FHIR R4 Resource Serialisation

**Problem**: All clinical records are stored and returned as opaque `dict[str, Any]`. Integration with external EMRs and HIEs requires proprietary adapters.

**Improvement**: Add `to_fhir()` / `from_fhir()` methods to each response model. Map `CarePlanResponse` → `CarePlan`, `HandoffResponse` → `Communication`, `CDSAlertResponse` → `DetectedIssue` using the FHIR R4 profile. Emit FHIR-compliant JSON from a `/fhir/` router prefix.

**Impact**: Zero-glue integration with HL7 FHIR-compliant EHR systems (Epic, Cerner, DHIS2).

---

## 3. Admission-to-Discharge Acuity Timeline

**Problem**: Clinical load is snapshot-based (dashboard_summary). Trend data is unavailable, making retrospective analysis impossible.

**Improvement**: Persist a time-series acuity record for each patient admission. Each state change (care plan status, CDS alert, workflow state transition) appends a timestamped entry. Expose a `patient_acuity_timeline()` method that returns the ordered event stream with severity scores for a given admission window.

**Impact**: Supports retrospective case review, M&M analysis, and outcome correlation with clinical interventions.

---

## 4. Constraint-Based Bed Management Integration

**Problem**: Discharge planning is isolated from bed occupancy. Planned discharge dates are not validated against bed turnaround requirements.

**Improvement**: Add a `bed_management_snapshot()` method that models occupancy, predicted discharge dates, and pending admission queue per unit. Expose a constraint-solving `optimise_discharge_schedule()` that proposes discharge order to minimise bed-days and flag delayed discharges breaching 4-hour targets.

**Impact**: Reduces average length of stay and unlocks bed capacity visibility for ward managers.

---

## 5. Clinical Documentation Quality Scorer

**Problem**: Free-text fields (situation, background, assessment, recommendation in SBAR) are accepted without quality validation. Poor documentation degrades handoff safety.

**Improvement**: Add a `score_documentation_quality()` method that checks SBAR completeness using structural heuristics (minimum field lengths, presence of medication names, allergy mentions, vital sign values). Return a numeric score 0–100 with per-field breakdown and specific remediation hints.

**Impact**: Directly reduces handoff-related adverse events by surfacing incomplete documentation before sign-off.

---

## 6. Protocol Deviation Detection and Auto-Alert

**Problem**: Protocols are activated but not monitored. Deviation (step skipped, step overdue by > N hours) is invisible until audit.

**Improvement**: Add a `detect_protocol_deviations()` background method that compares each active protocol's expected step schedule to actual completions. Emit a `protocol_deviation_detected` CDS alert automatically when a step is overdue by the configured threshold.

**Impact**: Converts protocol tracking from passive documentation to active safety enforcement.

---

## 7. Structured Consent Management

**Problem**: Consent is referenced in care plans by implication only. There is no explicit model or audit trail for informed consent.

**Improvement**: Add `ConsentRecord` model and `record_consent()` / `verify_consent()` / `withdraw_consent()` async methods. Track consent type (treatment, research, data sharing), version of consent form, witness, and withdrawal reason. Block care plan activation if required consent is absent.

**Impact**: Closes a medico-legal gap and satisfies regulatory requirements (Kenya Health Act, GDPR equivalents).

---

## 8. Multi-Factor Risk Stratification

**Problem**: Individual risk tools (fall, sepsis, nutrition, pain) exist in isolation. Composite risk is not surfaced.

**Improvement**: Add a `composite_risk_stratification()` method that aggregates Morse, MUST, qSOFA, and pain scores into a unified risk profile per patient visit. Apply weighted scoring and return a colour-banded risk tier (green/amber/red) with contributing factors ranked by impact.

**Impact**: Gives the bedside clinician a single-number summary instead of requiring manual triangulation across four separate assessments.

---

## 9. Outcome Tracking and Readmission Prediction

**Problem**: Care plan completion is recorded but clinical outcomes (mortality, readmission, complication) are not captured or analysed.

**Improvement**: Add `record_outcome()` for structured outcome entry (discharge status, complications, 30-day readmission flag) and `readmission_risk_score()` that applies LACE+ index logic (Length of stay, Acuity, Comorbidity, Emergency department visits) to return a probability score and recommended follow-up intensity.

**Impact**: Enables outcomes-based reporting required for value-based care contracts and JCI accreditation.

---

## 10. Clinical Task Escalation Engine

**Problem**: Overdue workflows are detected at query time only. No proactive escalation chain is triggered.

**Improvement**: Add an `escalate_overdue_tasks()` method that, when called periodically, identifies workflows overdue by configurable thresholds (1h, 4h, 24h), transitions them to `escalated` state, creates a linked CDS alert, and records the escalation path in the audit trail.

**Impact**: Converts passive overdue detection into an active escalation process, reducing task abandonment rates.

---

## 11. Antimicrobial Stewardship Tracker

**Problem**: Broad-spectrum antibiotic use recommended by the sepsis bundle is not tracked for duration or de-escalation review.

**Improvement**: Add `antimicrobial_prescription_register()` and `antimicrobial_review_due()` methods. Each antibiotic course started under a sepsis bundle is registered with start date and review trigger (48h for de-escalation, 72h for culture-guided switch). `antimicrobial_review_due()` returns courses requiring clinical review and suggests de-escalation based on organism sensitivity where available.

**Impact**: Directly reduces antimicrobial resistance propagation and meets WHO AMS programme requirements.

---

## 12. Structured Adverse Event Reporting

**Problem**: The M&M review process exists but there is no structured adverse event / near-miss intake form aligned to the WHO Adverse Event Severity Classification.

**Improvement**: Add `report_adverse_event()` with a structured intake model: event type, WHO severity grade (1–5), contributory factors, immediate actions taken, harm prevention flag. Integrate with `morbidity_mortality_review()` so severe events (grade ≥ 3) automatically enter the M&M queue.

**Impact**: Produces a systematic safety event registry suitable for regulatory reporting and systemic learning.

---

## 13. Pre-Operative Checklist Automation

**Problem**: Surgical safety checklists (WHO Surgical Safety Checklist) are managed on paper, preventing digital audit and completion tracking.

**Improvement**: Add `initiate_surgical_safety_checklist()` that generates a structured three-phase checklist (Sign In / Time Out / Sign Out) as a workflow set. Each phase requires explicit completion by the assigned role (anaesthetist, surgeon, scrub nurse). Incomplete phases block the `complete_protocol()` call for the associated surgical protocol.

**Impact**: Digitises the WHO SSC, enabling real-time compliance tracking and eliminating paper documentation gaps.

---

## 14. Care Bundle Compliance Dashboard

**Problem**: Individual bundle components (e.g., Sepsis-6, VTE prophylaxis bundle) are not tracked as a unit. Partial bundle completion is invisible.

**Improvement**: Add `track_care_bundle()` and `bundle_compliance_report()` methods. Each bundle is modelled as a named ordered checklist. Components can be ticked individually. The compliance report shows bundle completion rate, time from activation to each component, and identifies which components are most frequently missed.

**Impact**: Enables quality improvement teams to identify the highest-leverage intervention gaps within bundles.

---

## 15. Async Event Emission via CloudEvents

**Problem**: All state changes are recorded in `_audit_events` in-memory only. Downstream systems (EMR, lab, analytics) have no live feed.

**Improvement**: Refactor `_audit()` to emit CloudEvents-formatted messages onto an async message bus (configurable: Redis Streams, NATS, or local queue). Each event carries `source`, `type`, `subject`, `datacontenttype`, and a structured `data` payload. Add `replay_audit_events()` to rehydrate service state from the event log for event-sourcing scenarios.

**Impact**: Transforms the service from a closed in-process store to an event-producing participant in the APG integration mesh, enabling real-time dashboards, EMR synchronisation, and audit replays.
