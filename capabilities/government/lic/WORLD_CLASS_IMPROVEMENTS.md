# World-Class Improvements: Licensing & Permitting (government_lic)

**Capability**: `government_lic` | **Domain**: `government` | **Date**: 2026-06-11

## Improvement Catalogue

### 1. Risk-Based Inspection Scheduling
Replace the current random inspection selection with a risk-scoring model that weighs prior violation history, licence type hazard level, time since last inspection, and sector-specific compliance trends. High-risk licences are inspected at 2x the base rate. This reduces regulatory burden on compliant businesses while concentrating resources where harm is most likely.

### 2. Digital Licence Wallet (QR/NFC)
Issue each licence as a cryptographically signed digital credential (JSON-LD VC / W3C VC 2.0) accessible in a citizen wallet. Inspectors can verify authenticity offline via QR scan without an internet connection. Eliminates paper certificates and forgery. Composes with `government_csr` for portal delivery.

### 3. Predictive Expiry Engine
Train a lightweight gradient-boosted model on historical renewal patterns to predict which licences will lapse without proactive outreach. Fire multi-channel notifications (SMS, email, push) at the optimal lead-time per licence type rather than a static 30-day window. Reduces late-renewal penalties and increases on-time renewal rates.

### 4. Automated Fee Schedule Versioning
Store fee schedules as time-versioned records keyed by `(licence_type, effective_date)`. When a renewal or new application falls within a fee change window, the service applies the correct rate automatically and appends a fee_schedule_version to every FeeRecord. Removes manual fee adjustment errors during regulatory review cycles.

### 5. Multi-Jurisdiction Reciprocity Graph
Model jurisdiction-to-jurisdiction reciprocity agreements as a directed graph with agreement metadata (conditions, expiry, coverage). `inter_jurisdiction_check` traverses the graph transitively to detect indirect validity paths. Enables cross-border trade licence recognition without per-pair hard-coding.

### 6. Inspection Checklist Engine
Replace the hardcoded `["fire_safety", "sanitation", ...]` checklist with a configurable, versioned checklist registry keyed by `(inspection_type, licence_type)`. Inspectors complete digital checklists with weighted scoring; a minimum passing score gates renewal. Checklist versions are immutable once published so historical inspections remain reproducible.

### 7. Appeal & Tribunal Workflow
Add a formal appeal lifecycle for revocations and suspension decisions: `appeal_filed` → `tribunal_assigned` → `hearing_scheduled` → `decision_issued` → (`upheld` | `overturned`). Each transition emits an event to the `mqeb` stream. Composes with `government_cas` for case management and `audl` for decision audit.

### 8. Late-Fee Auto-Assessment
When a renewal is submitted after the expiry date, automatically compute and attach a `LicenceLateFeePenalty` record using the configurable `late_fee_rate_per_day` setting. The penalty is included in the fee reconciliation report and blocks issuance of the renewed licence until paid.

### 9. Parallel Document Verification
Replace the sequential document-check loop in `submit_application` with a `asyncio.gather` fan-out that verifies each document reference against the document store in parallel. Returns per-document status alongside the application receipt, cutting verification latency by up to 80% for applications with multiple attachments.

### 10. Condition-Based Licence Endorsements
Allow licences to carry typed conditions (e.g. `operating_hours`, `geographic_boundary`, `staff_certification_required`) as structured JSON rather than free-text notes. Condition violations detected during inspections are automatically linked to the parent licence and trigger a compliance event. Conditions are versioned and can be amended through a separate endorsement workflow.

### 11. Batch Application Processing via Event Stream
Integrate with the `mqeb` bytewax stream for high-volume batch intake: up to 10,000 applications per batch job, with per-record error isolation and a dead-letter queue for failures. Enables bulk licence revalidation during regulatory regime changes without API rate concerns.

### 12. Service-Level Agreement Tracking
Define SLA targets per licence type (e.g. 21 calendar days for business licences, 5 days for temporary event permits). Track actual processing time per application stage. Surface SLA breach warnings at T-3 days and escalate overdue applications to a supervisor queue automatically. Publish SLA metrics to the KPI dashboard.

### 13. Offline-First Mobile Inspector App Support
Expose an `inspection_sync` endpoint that packages all inspections scheduled for a given inspector as a signed payload for local SQLite. Completed inspection reports are queued locally and synced when connectivity is restored with idempotent conflict resolution. Eliminates inspection delays in areas with poor connectivity.

### 14. Regulatory Change Impact Analysis
Before any fee schedule or policy rule change is applied, run an `impact_analysis` dry-run that computes: number of licences affected, estimated revenue delta, and licences that would fail revalidation under the new rules. Produces a human-readable impact report for sign-off before the change takes effect.

### 15. Composable Compliance Score Card
Compute a per-licence `ComplianceScore` (0–100) aggregating: inspection pass/fail history (40%), fee payment timeliness (20%), renewal timeliness (20%), and condition adherence (20%). The score drives risk-based inspection frequency, fee discount eligibility for consistently compliant holders, and public-register trust badges. Score history is retained for trend analysis.
