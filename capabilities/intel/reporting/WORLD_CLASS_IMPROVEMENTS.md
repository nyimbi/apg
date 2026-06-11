# World-Class Improvements — Intelligence Reporting Capability

## Overview

These 15 improvements elevate `intel_reporting` from a functional prototype to a production-grade intelligence dissemination platform. Each addresses a concrete gap in reliability, observability, security, or analytical depth.

---

## 1. Event-Sourced Audit Trail with Tamper-Evidence

**Problem**: `audit_events` is a plain in-memory list — no ordering guarantees, no integrity proof, trivially modified.

**Improvement**: Append-only event log where each entry carries a SHA-256 hash of `(prev_hash, event_payload)`, forming a hash chain. On read, `verify_audit_chain()` recomputes and detects any tampering. Integrates with PostgreSQL `GENERATED ALWAYS AS` columns for persistent storage.

**Impact**: Satisfies auditor and regulator requirements for intelligence systems (e.g., IC ICD 711, GDPR Article 30).

---

## 2. Classification-Label Enforcement at Every Write Boundary

**Problem**: `normalize_code(classification)` lowercases and strips the value, but there is no enforcement that a child record's classification never exceeds the parent workspace or template classification.

**Improvement**: Introduce a `ClassificationLattice` that encodes the partial order `unclassified < restricted < confidential < secret < top_secret`. Every `record_*` call asserts `child_classification <= parent_classification`. Upgrades are only permitted with a recorded authority of type `classification_upgrade`.

**Impact**: Closes a critical data-spill vector; prevents accidental over-classification of low-side workspaces.

---

## 3. Pluggable Persistence Backend (Repository Pattern)

**Problem**: All state lives in `dict` attributes on the service instance — no persistence across process restarts, no horizontal scaling.

**Improvement**: Extract a `ReportingRepository` abstract base class with `async get`, `async put`, `async delete`, `async query` methods. Provide `InMemoryRepository` (current behavior), `PostgreSQLRepository` (asyncpg + SQLAlchemy 2.0 core), and `RedisRepository` (for hot-path reads). Wire via DI at `__init__` time.

**Impact**: Unlocks multi-process deployments, blue-green deploys, and DB-backed audit logs without touching service logic.

---

## 4. Report Versioning and Diff Engine

**Problem**: Once a section is added, there is no way to track what changed between versions — a hard requirement for intelligence products that undergo multiple editorial cycles.

**Improvement**: Implement `async version_report(report_id)` that snapshots the current product + sections + citations into an immutable `ReportVersion` object keyed by `(product_id, version_number)`. `async diff_versions(report_id, v1, v2)` returns a structured diff of sections added/removed/modified, with confidence-score deltas. Versions are stored in the repository backend.

**Impact**: Enables before/after audit during review, supports rollback to a prior approved version, and satisfies post-publication correction workflows.

---

## 5. Recipient Need-to-Know Validation Before Distribution

**Problem**: `disseminate_report()` distributes to any string in `distribution_list` with no validation that recipients hold clearances compatible with the report's classification.

**Improvement**: Introduce a `NeedToKnowRegistry` (backed by the `auth` capability adapter) that stores `(recipient_id, max_clearance, compartments)`. `disseminate_report` calls `registry.validate_recipient(recipient, report_classification, compartments)` and raises `PermissionError` for under-cleared recipients, logging each check to the audit trail.

**Impact**: Closes the most common classification-spill path in dissemination systems.

---

## 6. Structured Key Intelligence Questions (KIQ) Lifecycle

**Problem**: Analytic judgments and caveats are stored as free-text in `_report_feedback` with no linkage back to the intelligence requirements that prompted the report.

**Improvement**: Implement a `KIQ` model with fields `question`, `priority`, `status` (`open | answered | deferred`), and `answer_product_id`. New service methods: `async register_kiq(kiq_id, question, priority)`, `async answer_kiq(kiq_id, product_id)`, `async kiq_coverage_report()` that shows which KIQs are answered, deferred, or still open. Reports can be tagged with `kiq_ids` at creation time.

**Impact**: Closes the loop between intelligence requirements management and reporting production — a cornerstone of the intelligence cycle.

---

## 7. Source Reliability and Information Credibility (SRIC) Scoring

**Problem**: `confidence_score` on a section is a naive proxy derived from content length. It carries no intelligence-community semantics.

**Improvement**: Implement a standard 6×6 SRIC matrix (NATO STANAG 2022 / Admiralty Scale) as an enum pair `(SourceReliability: A-F, InformationCredibility: 1-6)`. `record_section` accepts `sric: tuple[str, int]` instead of (or alongside) `confidence_score`. `intelligence_score()` maps SRIC pairs to a numeric score using the published lookup table.

**Impact**: Produces machine-readable reliability assessments that are interoperable with allied intelligence systems.

---

## 8. Parallel Dissemination with Structured Failure Handling

**Problem**: `disseminate_report` iterates recipients sequentially. A single slow or failing recipient blocks all others and leaves the report in an inconsistent partially-disseminated state.

**Improvement**: Fan out distribution records via `asyncio.gather(*tasks, return_exceptions=True)`. Introduce a `DistributionResult` model with `status: success | pending | failed`, `retry_count`, and `last_error`. A `async retry_failed_distributions(product_id)` method retries only failed records. State machine transitions only to `REPORT_DISSEMINATED` when all mandatory recipients succeed.

**Impact**: Eliminates silent partial disseminations that are invisible to operators today.

---

## 9. Redaction Engine for Downgraded Copies

**Problem**: There is no way to produce a lower-classification copy of a report for wider distribution without manually re-authoring it.

**Improvement**: Implement `async redact_report(source_product_id, target_classification, redaction_authority_id)` that clones the product at a lower classification, removes sections whose `classification` exceeds the target, replaces them with `[REDACTED]` placeholders, and creates a new product linked to the source via `parent_product_id`. The entire operation is gated by a recorded authority of type `classification_downgrade`.

**Impact**: Supports the "write for release" and "tearline" workflows standard in FVEY intelligence production.

---

## 10. Machine-Readable Report Schema Registry

**Problem**: `template_reference` is an opaque string. There is no enforcement that sections conform to the structure the template implies.

**Improvement**: Templates carry a JSON Schema payload (`schema_definition: dict`). `record_section` validates `section_reference` content against the schema for the template's section type. `async validate_report_structure(product_id)` runs the full structural check and returns a report of validation errors per section.

**Impact**: Prevents malformed reports from reaching the approval queue, catching structural errors at authoring time rather than reviewer time.

---

## 11. Metrics Emission via OpenTelemetry

**Problem**: There are no metrics exposed for operational monitoring — no request latency, throughput, error rates, or queue depth signals.

**Improvement**: Instrument `IntelligenceReportingService` with OpenTelemetry SDK: span per public method, counters for `reports_created_total`, `distributions_sent_total`, `approval_rejections_total`, and a histogram for `report_lifecycle_duration_seconds` (time from `draft` to `disseminated`). Expose via OTLP exporter, configurable by env var.

**Impact**: Enables SLA dashboards, alerting on approval queue buildup, and performance regression detection.

---

## 12. Async Background Classification Review Scheduler

**Problem**: Reports can sit in `REPORT_PEER_REVIEW` indefinitely. There is no SLA enforcement or escalation mechanism.

**Improvement**: Implement `async schedule_review_sla(product_id, sla_hours)` that stores an SLA deadline. A background `async enforce_review_slas()` coroutine (runnable via Bytewax or `asyncio.create_task`) scans overdue reviews, fires escalation events via the `notify` adapter, and marks reviews as `escalated` in the state machine.

**Impact**: Operationalizes time-sensitive intelligence dissemination requirements (e.g., operational reporting SLAs).

---

## 13. Compartment and Codeword Management

**Problem**: `classification` is a single string — there is no support for SCI compartments, SAP codewords, or REL-TO markings that are fundamental to modern intelligence security frameworks.

**Improvement**: Extend `ReportingProduct` and `ReportingSection` with `compartments: list[str]` and `caveats: list[str]` (e.g., `["NOFORN", "REL TO GBR"]`). The `NeedToKnowRegistry` checks compartment membership. `record_product` and `record_section` validate compartment strings against a `CompartmentRegistry` seeded from the `authority` capability.

**Impact**: Makes the capability usable for real classified intelligence environments, not just unclassified proxies.

---

## 14. Natural Language Summary Generation via Local LLM

**Problem**: Analysts must manually write executive summaries. There is no automation hook for generating first-draft summaries from section content.

**Improvement**: Implement `async generate_executive_summary(product_id, model: str = "mistral:7b")` that concatenates `section_reference` fields, sends them to a locally hosted Ollama model, and stores the result as a new section of type `executive_summary` with `confidence_score=0.6` (pending analyst review). The call is gated by `human_approval_recorded=True` in `validate_agent_action`.

**Impact**: Reduces analyst load on routine reporting; aligns with the project's locally hosted open-source AI strategy.

---

## 15. Report Subscription and Change Notification

**Problem**: Consumers of intelligence products have no way to subscribe to updates — they must poll `report_index()` manually.

**Improvement**: Implement a `SubscriptionRegistry` with `async subscribe(subscriber_id, filters: dict)` (filter by `classification`, `product_type`, `kiq_id`). Lifecycle state transitions (`draft → peer_review → approved → disseminated`) emit events to matching subscribers via the `notify` adapter. `async list_subscriptions(subscriber_id)` and `async unsubscribe(subscriber_id, subscription_id)` round out the API.

**Impact**: Enables push-based intelligence dissemination and real-time consumer awareness — critical for time-sensitive operational intelligence.
