# pharma_rec — World-Class Improvement Plan

**Capability**: Records Management (`pharma_rec`)
**Domain**: Pharmaceutical Regulatory Compliance
**Date**: 2026-06-11
**Author**: Nyimbi Odero

---

## 1. Electronic Signature (21 CFR Part 11 / EU Annex 11) Enforcement

Current QP approval is a boolean flag with no cryptographic binding. Replace with an e-signature record: signer identity, timestamp, IP address, reason-for-signing, and HMAC over the document hash. Reject approval workflows where the manifest is incomplete or the signer's credential has expired. This is the single most dangerous gap for FDA inspection readiness.

## 2. Document Version Graph with Immutable Audit Spine

Labels and SOPs undergo serial revisions but the current store is a flat dict — supersession relationships exist only as a nullable string field. Model the version history as a directed acyclic graph stored in PostgreSQL using a recursive CTE. Each node carries a content hash; the graph structure is append-only. Mutations produce new nodes, never overwrite existing ones. Satisfies 21 CFR Part 11 §11.10(e) audit trail requirements.

## 3. Training Records Module with Competency Matrix

Regulatory SOPs require documented training before personnel may perform covered procedures. Add a `TrainingRecord` model and `TrainingCurriculumService` that maps roles to required SOPs, tracks completion dates and assessment scores, and blocks personnel from performing regulated activities until their training is current. Emit `training_overdue` events to the event bus.

## 4. CAPA Bidirectional Linkage

Inspection findings and gap assessments currently produce audit events but do not create CAPA records in `pharma_qms`. Add an async `raise_capa()` method that calls the `pharma_qms` capability contract, records the returned CAPA ID back on the finding, and tracks CAPA closure status. Without this link, inspection readiness dashboards show incomplete remediation state.

## 5. Structured Inspection Readiness Checklist Engine

The `InspectionRecord` model has no pre-inspection readiness state. Add a `ReadinessChecklist` model with configurable checklist templates per inspection type (GMP, GCP, GDP, ISO 13485). Each checklist item tracks owner, due date, evidence reference, and completion status. Gate `record_inspection_outcome()` on checklist completion percentage exceeding a configurable threshold.

## 6. Variation Filing Integration with `pharma_reg`

Label changes trigger Type IA/IB/II variation filings in the EMA/FDA submission system. The current `approve_label()` method has no outbound integration. Add `async file_variation()` that invokes the `pharma_reg` capability contract, attaches the returned submission number to the `LabelRecord`, and listens for approval/rejection events via the message queue.

## 7. Real-Time Commitment Risk Scoring

Overdue detection is binary and runs on-demand. Replace with a continuous risk score: `days_remaining / total_duration` weighted by commitment criticality and authority power level (FDA > EMA > local NCA). Persist the score and emit `commitment_risk_changed` events when the score crosses configurable thresholds. Expose a ranked risk queue on the dashboard.

## 8. Regulatory Intelligence NLP Pipeline

Guidance documents and regulations arrive as free text. Add an async `classify_intel()` method that calls the local Ollama LLM (via `nlpc` capability) to: (a) extract affected product classes, therapeutic areas, and impacted dossier sections; (b) assign a structured impact tag (labeling, manufacturing, clinical, non-clinical, CMC); (c) auto-route to the correct product owners. Reduces manual triaging from days to minutes.

## 9. Authority Interaction Minutes & Action Item Tracker

`authority_interaction()` records meetings but does not capture structured minutes, open questions, or follow-up deadlines. Add a `MeetingMinutes` model with line-by-line action items, owners, and due dates. Emit `action_item_created` events and integrate with `wflo` for automated reminders. This is auditable evidence required by EMA procedural guidance.

## 10. Import Licence Automated Renewal Workflow

`import_licence()` creates a single record with no renewal logic. Licences expire and require re-application typically 60–90 days before expiry. Add state-machine transitions (applied → issued → active → near_expiry → renewal_submitted → renewed) and an async `check_licence_renewals()` cron method that scans all active licences, calculates days to expiry, triggers renewal workflows, and blocks shipment releases when a licence lapses.

## 11. Regulatory Submission Dossier Completeness Check

Before filing a variation or new application, a dossier completeness gate validates required CTD modules, mandatory templates, and format rules (eCTD v3.2.2 / NeeS). Add `async validate_dossier(dossier_id, submission_type)` that applies a rule engine over the attached documents, returns a structured completeness report with per-section pass/fail, and blocks submission until critical gaps are resolved.

## 12. Multi-Jurisdiction Parallel Compliance Tracking

A single product can have simultaneous compliance obligations across FDA, EMA, Health Canada, PMDA, TGA, and local NCAs. The current data model uses `region` as a string field with no normalization. Introduce a `JurisdictionProfile` model that maps each jurisdiction to its specific regulatory framework, renewal cycle, reporting calendar, and authority contacts. Compliance calendars then aggregate across all jurisdictions per product, surfacing cross-border conflicts (e.g., a US PSUR period conflicting with EU PBRER cycle).

## 13. Batch Record Integration for Recall Readiness

During inspections, authorities frequently request batch traceability: which batch was released, by which QP, under which label version, to which markets. Add `async trace_batch(batch_id)` that joins batch release records from `pharma_qms`, label records from `pharma_rec`, and distribution records from `pharma_sup` into a single traceability bundle. The result must be retrievable within 10 minutes — a hard SLA for recall scenarios.

## 14. Configurable Escalation Matrix via Event Bus

Currently all escalations write audit events to an in-memory list. Replace with a proper escalation matrix: each event type maps to a recipient role, escalation delay, and channel (email, Slack, SMS). Implement `async dispatch_escalation(event_type, reference_id, tenant_id)` backed by the `ntfy` capability. Configuration is tenant-scoped and hot-reloadable from the `conf` capability without service restart.

## 15. Compliance Health Score with Benchmarking

Synthesize a single numeric compliance health score (0–100) per tenant-product pair using weighted KPIs: inspection outcomes (30%), open commitments risk (25%), label currency (15%), PMS status (15%), intel response time (10%), training coverage (5%). Persist historical scores to enable trend analysis. Expose via `async compliance_health_score(product_id, tenant_id)` and the dashboard. Benchmarking against anonymised peer data (where consent exists) enables gap prioritisation relative to industry norms.
