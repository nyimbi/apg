# World-Class Improvements: pharma_com — Commercial Operations

**Capability**: Pharmacovigilance / Commercial Operations (`pharma_com`)
**Domain**: `pharma`
**Author**: Nyimbi Odero — Datacraft
**Date**: 2026-06-11

---

## 1. True Async Service Layer

All public methods are currently synchronous despite the service using async stubs at the bottom. The entire service must be ported to `async def` with `await`-able I/O gates (DB, audit bus, notification adapter). Blocking dict lookups mask future PostgreSQL calls; the coroutine boundary must be established now so callers never need to change their `await` sites.

**Impact**: Eliminates thread-pool pressure when running under ASGI (Starlette, FastAPI). Critical before PostgreSQL migration.

---

## 2. PostgreSQL-Backed Persistence via Async SQLAlchemy

In-memory `dict` stores are single-process and ephemeral. Replace with `asyncpg` + `SQLAlchemy 2.x` async session factory injected at construction. The existing Alembic migration scaffolding (`alembic/versions/0001_initial.py`) is already present — wire it up. Use `AsyncSession` with explicit `expire_on_commit=False`.

**Impact**: Production-ready durability; horizontal scaling across workers.

---

## 3. Event Streaming via Bytewax / Kafka

The `_audit` helper appends to an in-process list. Replace with a proper event bus: emit domain events (`territory_created`, `sample_dispensed`, `pdma_violation_detected`) to a Kafka topic / Bytewax dataflow. This enables downstream caps (`pharma_rec`, `grc`, `comp`) to subscribe without polling.

**Impact**: Real-time compliance signal propagation; decoupled downstream consumers.

---

## 4. Pharmacovigilance Signal Detection

The capability description mentions signal detection but no implementation exists. Add `detect_adverse_event_signals()` using disproportionality analysis (Reporting Odds Ratio, PRR) over the `_calls` and `_interactions` corpus. Flag product-event pairs that breach configurable thresholds, and route findings to `pvi` capability.

**Impact**: Delivers the "signal detection" pillar from the DESC — currently completely absent.

---

## 5. Regulatory Submission Workflow (E2B R3 / CIOMS)

No submission pipeline exists. Add an `initiate_regulatory_submission()` method that packages adverse event data into ICH E2B(R3) XML or CIOMS I form, validates mandatory fields, attaches to a submission record, and tracks submission status (draft → submitted → acknowledged → closed). Integrate with EMA EVDAS and FDA FAERS gateway adapters.

**Impact**: Closes the "regulatory submissions" gap entirely; aligns with ICH E2B(R3) mandate.

---

## 6. Adverse Event Report (ICSR) Lifecycle

Add full Individual Case Safety Report (ICSR) management: `create_icsr()`, `update_icsr()`, `submit_icsr()`, `acknowledge_icsr()`. ICSRs must track: patient demographics, reporter information, suspect products, adverse reactions (MedDRA coded), causality assessment, and seriousness criteria. Store with immutable audit history.

**Impact**: Core pharmacovigilance record — cannot operate PV without it.

---

## 7. MedDRA Terminology Integration

Adverse event coding currently uses free-text `notes`. Add a `MedDRAService` adapter that maps verbatim terms to MedDRA PT/LLT/HLT/SOC hierarchy. Support fuzzy matching via local embeddings (Ollama-served model) or deterministic dictionary lookup against a loaded MedDRA release. Expose `encode_meddra_term()` and `decode_meddra_code()`.

**Impact**: Regulatory submissions require MedDRA coding; this is non-negotiable for PV systems.

---

## 8. Sunshine Act / Open Payments Reporting Pipeline

The `record_spend` method enforces caps but does not produce Open Payments-format export. Add `generate_open_payments_report()` that aggregates spend by HCP NPI for a calendar year, validates mandatory fields (NPI, spend type, amount), and exports as CMS Open Payments XML/CSV. Enforce deduplication of records before submission.

**Impact**: US regulatory mandate — penalties up to $1M/year for non-compliance.

---

## 9. Risk-Based Signal Prioritisation (Triage Score)

Add `compute_signal_triage_score()` that combines: reporting frequency, severity distribution, novelty (not in approved label), product age on market, and geographic clustering. Output a composite score (0–100) with a tier label (`watch` / `investigate` / `escalate`). Feed to `intel` capability for dashboard display.

**Impact**: Transforms raw signal counts into actionable priority queue; reduces reviewer burden.

---

## 10. Multi-Level Approval Chain with Deadlines

The `approve_plan()` method accepts a single approval reference with no workflow state machine. Replace with a configurable N-level approval chain: each level has an assignee, deadline, and escalation rule. Track `pending_approver`, `escalated_at`, `approved_at` per level. Emit `approval_overdue` events to `ntfy`.

**Impact**: Meets SOX / pharma SOPs that mandate multi-level sign-off with audit trails.

---

## 11. Sample Cold-Chain Integrity Tracking

`SampleDispensing` records lot/expiry but not temperature excursion data. Add `record_cold_chain_event()` that logs temperature readings against a sample lot, evaluates against product-specific thresholds, flags excursions, and blocks dispensing of affected lots. Integrate IoT sensor data via MQTT adapter.

**Impact**: GxP requirement; excursion-affected samples dispensed to patients create liability.

---

## 12. CAPA (Corrective and Preventive Action) Integration

Compliance violations (PDMA breach, aggregate cap exceeded) currently raise `PermissionError` and stop. They should also spawn a CAPA record routed to `qms` capability: capture root cause, corrective action, responsible person, and due date. Add `create_capa_from_violation()` and `close_capa()`.

**Impact**: Closed-loop quality management; FDA Form 483 responses require CAPA evidence.

---

## 13. Physician Prescription Trend Analytics (Rx Lift)

`prescriber_analytics` reports call coverage but not Rx impact. Add `compute_rx_lift()` that ingests prescription volume data (via `pharma_rec` adapter), correlates call activity with Rx trend change pre/post visit, and computes incremental lift per rep/territory/product. Output p-value and confidence interval.

**Impact**: ROI quantification for field force investment — the primary commercial KPI.

---

## 14. Duplicate Detection and Deduplication Engine

ICSRs and spend records frequently arrive as duplicates from multiple sources (EHR, spontaneous reports, literature). Add `detect_duplicates()` using configurable match keys (patient DOB, initiation date, suspect drug, reaction PT code) and a probabilistic deduplication score. Allow merge of duplicate ICSRs with full lineage tracking.

**Impact**: ICH E2B guidance requires deduplication before regulatory submission; duplicates inflate signal statistics.

---

## 15. Tenant-Configurable Compliance Rule Engine

Hardcoded thresholds (`aggregate_cap = 500.0`, `receipt_required_above = 25.0`) scattered in method bodies are unmaintainable across jurisdictions (EU vs US vs APAC). Extract all thresholds to a `ComplianceRuleSet` Pydantic model loaded from the `conf` capability at service init. Support per-tenant, per-product, and per-country rule overrides with effective-date versioning.

**Impact**: Single-tenant config changes currently require code deployment; this unblocks multi-market rollout.

---

## Summary Priority Matrix

| # | Improvement | Effort | Compliance Risk if Missing | Priority |
|---|-------------|--------|--------------------------|----------|
| 6 | ICSR Lifecycle | High | Critical | P0 |
| 5 | Regulatory Submission (E2B R3) | High | Critical | P0 |
| 7 | MedDRA Integration | Medium | Critical | P0 |
| 4 | Signal Detection | Medium | High | P1 |
| 8 | Sunshine Act Export | Medium | High | P1 |
| 15 | Configurable Rule Engine | Low | High | P1 |
| 2 | PostgreSQL Persistence | High | Medium | P1 |
| 1 | True Async Service | Medium | Medium | P2 |
| 3 | Event Streaming | Medium | Medium | P2 |
| 12 | CAPA Integration | Medium | Medium | P2 |
| 10 | Multi-Level Approval | Medium | Medium | P2 |
| 9 | Signal Triage Score | Low | Low | P3 |
| 13 | Rx Lift Analytics | Medium | Low | P3 |
| 14 | Deduplication Engine | High | Medium | P3 |
| 11 | Cold-Chain Tracking | High | Medium | P3 |
