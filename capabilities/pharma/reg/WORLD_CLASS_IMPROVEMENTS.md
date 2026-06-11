# Pharma Regulatory Affairs (pharma_reg) — World-Class Improvements

**Capability**: `pharma_reg` | **Domain**: `pharma` | **Version target**: 2.0.0

---

## 1. Async-First Service Layer

All synchronous methods in `service.py` should be converted to `async def`. The current mix of sync and async methods creates inconsistent call patterns and blocks the event loop when used in async contexts (FastAPI, LangGraph agents, Bytewax pipelines). A clean `async def` surface with `await` internally allows straightforward concurrency via `asyncio.gather` for multi-region submissions.

---

## 2. Persistent Storage via Async SQLAlchemy

In-memory dicts (`self._registrations`, `self._dossiers`, etc.) are wiped on process restart and cannot be shared across workers. Replace with an async SQLAlchemy session scoped to the tenant, backed by PostgreSQL. This gives ACID guarantees, full-text search on dossier content, and proper foreign key constraints across registration → dossier → variation chains.

---

## 3. Strongly-Typed Variation Types

`RegistrationVariation.variation_type` has a validator accepting `["variation_type_ia", "variation_type_ib", "variation_type_ii", "extension"]` but `variation_application()` accepts `{"type_ia", "type_ib", "type_ii", "extension", "transfer", "line_extension"}`. These two sets are out of sync. Introduce a `VariationType` `StrEnum` shared by the model and service to enforce a single source of truth and fail loudly at model construction, not at policy enforcement.

---

## 4. Structured Regulatory Calendar with Deadline Engine

Regulatory authorities have statutory review timelines (e.g. EMA: 210-day clock, FDA PDUFA: 12 months, TFDA: 90 days). The service has no concept of review deadlines. Add a `DeadlineEngine` that tracks clock start/stop events (response to Day 120 LoQ pauses clock), computes days-remaining against authority-specific SLAs, and emits `deadline_approaching` events at configurable thresholds. This is essential for regulatory affairs teams who manage parallel submissions across regions.

---

## 5. eCTD Structural Validation Beyond Flag

`validate_ectd()` sets `ectd_validated = True` unconditionally — it does not inspect actual dossier content. Integrate a lightweight eCTD structural validator (checking module folder/file naming conventions per ICH M8) and return a structured `EctdValidationReport` with per-module findings, sequence number correctness, and STF (Study Tagging File) presence. Soft warnings vs. hard errors should be distinguished.

---

## 6. Multi-Tenant Isolation at Query Level

All list/filter methods iterate `self._registrations.values()` and filter by `tenant_id` in Python. With thousands of registrations across many tenants this is O(N) per query. Partition storage by tenant at the key level or, better, enforce tenant scoping in the DB query layer so cross-tenant data never even reaches the application tier. Critical for SaaS deployments with strict data residency requirements.

---

## 7. Regulatory Intelligence Feed Integration

Pharma regulatory affairs teams monitor agency news feeds (EMA EPAR updates, FDA Orange Book deltas, WHO prequalification list). Add an `ingest_regulatory_intelligence(source, payload)` method that parses structured feeds, matches against the tenant's registered products by INN/brand name, and creates `AuthorityInteraction` records of type `regulatory_intelligence_update`. This closes the loop between market intelligence and submission planning.

---

## 8. PSUR / PBRER Lifecycle Tracking

Periodic Safety Update Reports (PSUR) and Periodic Benefit-Risk Evaluation Reports (PBRER) are mandatory post-approval obligations with authority-set data lock dates. The current model has no concept of PSUR schedules. Add a `PsurSchedule` model with `data_lock_date`, `submission_due`, `submitted_at`, `assessment_outcome`, linked to `ProductRegistration`. The `check_renewal_alerts` method should also surface overdue PSURs.

---

## 9. Dossier Version Control with Diff Tracking

`RegistrationDossier.version` is a free-form string (`"1.0"`) with no history. Regulatory dossiers go through multiple versions (original submission, responses to queries, post-approval updates). Add a `DossierRevision` model tracking `parent_version_id`, `change_summary`, `changed_modules`, `changed_by`, and `sequence_number`. `prepare_dossier()` should create revision 0; subsequent updates create child revisions with full lineage.

---

## 10. Automated Gap Analysis Against Region-Specific Requirements

Different regions have different CTD module requirements (ACTD for ASEAN omits parts of Module 1; ANVISA requires Módulo 1 in Portuguese). A `gap_analysis(dossier_id, target_region)` method should compare the dossier's `modules_present` and `format` against a machine-readable region requirement matrix and return a prioritised list of gaps with remediation actions. Currently `dossier_completeness_check` is region-agnostic.

---

## 11. Parallel Multi-Region Submission Orchestration

`bulk_submit_registrations` calls `submit_registration` sequentially in a `for` loop. For a product being registered in 12 markets simultaneously this is slow and lacks per-region failure isolation. Refactor to use `asyncio.gather(*[submit(...) for region in regions], return_exceptions=True)` and return a structured result with per-region success/failure status, so a failure in Kenya does not block submission to Uganda.

---

## 12. Condition-of-Approval Obligation Tracker

`conditions_of_approval: list[str]` stores conditions as unstructured strings. Approved products often carry binding post-approval commitments (e.g. "submit Phase IV study by 2027-03-31"). Add a `PostApprovalObligation` model with `obligation_text`, `due_date`, `status` (`pending|fulfilled|overdue`), `evidence_reference`. Surface overdue obligations in `dashboard_summary` and trigger `obligation_overdue` events.

---

## 13. Rollback / Saga Support for Multi-Step Workflows

The submit → query → response → approval flow mutates state across multiple methods with no compensation logic. If `registration_approval` fails mid-way (e.g. certificate storage fails), the registration is left in an inconsistent state. Implement a lightweight saga pattern: each step emits a domain event, and a `SagaCoordinator` can replay or compensate. At minimum, wrap multi-step mutations in a context manager that records pre-state and restores on exception.

---

## 14. Role-Based Permission Granularity in `_enforce`

`_enforce` calls `evaluate_capability_rules(context)` but the context dict is manually constructed at each call site with varying keys, making it easy to silently omit a required check (e.g. `qp_signed_off` is only checked in some code paths). Replace ad-hoc context dicts with typed `EnforceContext` Pydantic models per operation, validated at construction. This makes omitted security checks a type error rather than a silent policy bypass.

---

## 15. Structured Audit Log with Retention Policy and Replay

`_audit_events` is an in-memory list with no retention, no schema enforcement, and no replay capability. Replace with a structured `AuditEvent` Pydantic model including `event_id`, `tenant_id`, `actor_id`, `event_type`, `entity_type`, `entity_id`, `before_state`, `after_state`, `timestamp`, `ip_address`, `session_id`. Persist to a dedicated immutable audit table (append-only, no UPDATE/DELETE). Support replay for regulatory inspection queries covering the last 7 years (EU GMP Annex 11 requirement).

---

*Generated: 2026-06-11 | Author: Nyimbi Odero | © 2025 Datacraft*
