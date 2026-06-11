# DLPD World-Class Improvements

**Capability**: Data Loss Prevention (dlpd)
**Author**: Nyimbi Odero — Datacraft © 2025

---

## 1. Async-Native Service Core

All service methods are synchronous — blocking under I/O. Refactor `DlpdService` to `AsyncDlpdService` with `async def` throughout, using `asyncio.gather` for fan-out operations (bulk scans, multi-policy evaluation, incident fan-out notifications). The sync wrappers can delegate via `asyncio.run` for backward compatibility. Priority: eliminates GIL contention when the service runs inside async web frameworks (FastAPI, ASGI Flask).

---

## 2. Streaming Egress Inspection via AsyncGenerator

Large file/email scans currently buffer the entire content string. Replace with `async def inspect_egress_stream(content_stream: AsyncIterator[bytes])` that chunks content, hashes incrementally (rolling SHA-256), and emits partial classification decisions as they arrive. This cuts peak memory for large exports from O(N) to O(chunk_size) and enables early-abort on first critical hit.

---

## 3. Persistent Backend Adapters via Repository Pattern

In-memory dicts are not production-grade. Define abstract `PolicyRepository`, `ClassifierRepository`, `InspectionRepository`, etc. with async `get`, `put`, `delete`, `list_tenant` methods. Ship a `PostgresRepository` (asyncpg) and a `RedisRepository` (aioredis) implementation. The service holds repository references, not raw dicts. Zero logic change in the core; just swap backends at construction time.

---

## 4. Real-Time Classifier Confidence Calibration

`detect_classifier_hits` returns fixed confidence values. Replace with a calibrated scorer: store hit counts and false-positive feedback per pattern, then compute a Platt-scaled confidence as `P(TP | hit) = sigmoid(a * raw_score + b)` where `a`, `b` are fitted from the feedback store. `ml_classifier_train` should write these calibration parameters rather than just counting samples. Precision from `policy_effectiveness` then becomes a live posterior rather than a point estimate.

---

## 5. Risk-Score Aggregation Across Channels (Cross-Channel Correlation)

Current analytics aggregate per-channel independently. Implement `cross_channel_risk_profile(tenant_id, subject_id)` that joins egress inspections, shadow IT detections, cloud activity events, and endpoint events for a given user/device, computes a composite risk score (weighted sum over severity ordinals), and flags users whose 7-day rolling score exceeds a configurable threshold. This is the core signal for User and Entity Behavior Analytics (UEBA).

---

## 6. Policy Version History and Rollback

`update_policy` mutates the model in place with no history. Store an immutable `PolicyVersion` record on each mutation (copy-on-write). Expose `list_policy_versions(policy_id, tenant_id)` and `rollback_policy(policy_id, version_id, actor)`. This enables audit-clean configuration management required by SOC 2 / ISO 27001 change-management controls.

---

## 7. Classifier Hot-Reload Without Service Restart

Pattern keys are resolved at classify time, but regex compilation is not cached. Introduce a `PatternCache` that compiles regexes on first use (keyed by `(tenant_id, pattern_id, regex_hash)`) and invalidates on `regex_pattern_library` writes. Expose `reload_classifiers(tenant_id)` as an async admin method that flushes stale compiled patterns. Eliminates repeated re-compilation under high inspection throughput.

---

## 8. Structured Notification Dispatch

`DlpIncident` records `notifications_sent=True` but never actually dispatches anything. Implement `async def dispatch_incident_notifications(incident_id, tenant_id)` backed by a `NotificationAdapter` interface with concrete implementations for email (SMTP/sendgrid), Slack webhook, PagerDuty events API, and a no-op stub for tests. The service calls this automatically after `_open_incident` when the adapter is configured.

---

## 9. Tokenization and Redaction Engine

After classification, consumers often need the content with sensitive spans replaced. Add `async def redact_content(tenant_id, content, classifier_ids, mode)` where `mode` is `"mask"` (replace with `***`), `"tokenize"` (replace with a reversible vault token), or `"hash"` (one-way). The vault for tokenization should integrate with the `encr` adapter. This turns DLPD from a detection-only capability into an active data sanitization pipeline.

---

## 10. Legal Hold Lifecycle with Custodian Management

Legal hold is currently a boolean on `QuarantineItem`. Model it as a first-class `LegalHold` entity with `hold_id`, `case_reference`, `custodians` list, `placed_by`, `placed_at`, `scope` (list of item IDs or a classifier predicate), and `status` (`active`, `released`). Expose `place_legal_hold`, `add_custodian`, `release_legal_hold` (requires all-custodian sign-off), and `legal_hold_inventory` report. Required for e-discovery workflows and GDPR litigation exemptions.

---

## 11. Composite Content Inspection (Structured Data)

`scan_file` treats content as a flat string. For structured formats (JSON, CSV, SQL result sets), implement `async def scan_structured(tenant_id, data, schema_hint, policy_id, actor)` that walks field paths, applies per-field classifiers based on field name heuristics (e.g., `ssn`, `credit_card`, `email`), and returns a per-field sensitivity map. This is essential for database-export DLP where most content is benign but isolated columns are sensitive.

---

## 12. Automated False-Positive Remediation Loop

When `false_positive_feedback` accumulates N reports for the same pattern/classifier, automatically suppress that hit for that tenant (not globally) by inserting a tenant-local suppression entry and lowering the classifier's calibrated confidence. Expose `suppression_report(tenant_id)` to make the suppression inventory auditable. This closes the feedback loop without manual tuning intervention.

---

## 13. Differential Privacy Noise Injection for Analytics

`dlp_analytics` and `reporting_export` return exact counts. For multi-tenant SaaS deployments where the analytics layer is shared, inject Laplace noise (`ε`-differential privacy) on aggregate counts before returning them to non-admin callers. Expose `set_analytics_privacy_budget(tenant_id, epsilon)` and document the privacy/utility tradeoff. Prevents reconstruction of individual inspection records from repeated aggregate queries.

---

## 14. OpenTelemetry Instrumentation

No distributed traces, metrics, or spans are emitted. Instrument every public service method with `opentelemetry.trace.get_tracer(__name__)` spans annotated with `tenant_id`, `policy_id`, `severity`, `action`, and `content_hash` (never raw content). Export inspection latency histograms and incident rate counters to OTLP. This enables SLO tracking, anomaly alerting on DLP latency spikes, and capacity planning.

---

## 15. Graph-Based Policy Conflict Detection

Multiple active policies for a tenant can produce contradictory decisions for the same content (one says `allow`, another says `block`). Implement `async def detect_policy_conflicts(tenant_id)` that builds a policy-classifier bipartite graph, finds overlapping classifier sets across policies with differing `default_action` values, and returns a conflict report with suggested resolutions (e.g., "policy A and B share classifier `pii-email`; A allows, B blocks — recommend B takes precedence"). Surfaces before go-live rather than at inspection time.
