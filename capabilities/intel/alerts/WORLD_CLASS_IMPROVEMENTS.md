# Intelligence Alerts — World-Class Improvement Opportunities

**Capability**: `intel_alerts` | **Domain**: `intel` | **Version target**: 2.0.0

---

## 1. Async-First Architecture

All service methods are currently synchronous. Every I/O-touching path (DB writes, outbound webhooks, event-bus publishes) blocks the event loop when integrated into an async FastAPI or Starlette runtime. Replacing the in-memory dict store with an async repository layer (e.g. async SQLAlchemy + asyncpg) and wrapping all public methods with `async def` removes this bottleneck entirely.

**Impact**: 10–50x throughput improvement on I/O-bound workloads.

---

## 2. Persistent Storage Backend via Repository Pattern

The current service holds all state in plain Python dicts. This is a design-time convenience, not a production runtime. Introducing an `AlertRepository` abstract base class with a `PostgresAlertRepository` implementation lets the service be tested with in-memory fakes while production deployments use a durable, queryable store with ACID guarantees.

**Impact**: Survivable across restarts; supports multi-process/multi-node deployments.

---

## 3. Event-Driven Side Effects via CloudEvents / Bytewax

State mutations (acknowledge, resolve, escalate) currently trigger only an in-process audit log append. Emitting standardised CloudEvents on each mutation to a Bytewax stream enables downstream consumers (notification dispatchers, SIEM forwarders, ML pipelines) to react in real time without polling.

**Impact**: Decouples alerting core from notification and analytics concerns.

---

## 4. Threshold-Based Auto-Escalation Rules Engine

There is no automatic escalation path when an alert ages past its SLA without acknowledgement. A background scheduler (APScheduler or Celery Beat) evaluating `_SLA_MINUTES` thresholds against open alert ages and triggering `record_escalation` automatically would implement policy-driven escalation without human intervention for each alert.

**Impact**: Dramatically reduces mean time to escalate (MTTE) for high-severity alerts.

---

## 5. ML-Powered Anomaly Scoring for Signals

`record_signal` accepts a `confidence_score` as a raw float but does no scoring of its own. Integrating a locally-hosted Ollama model (e.g. `llama3:8b` or a fine-tuned classifier) for signal enrichment scoring — called asynchronously during `signal_enrichment` — gives analysts a calibrated risk score derived from contextual features, not just the upstream provider's raw value.

**Impact**: Reduces analyst cognitive load; improves true positive rate over time via feedback loop.

---

## 6. Deduplication Window Using Probabilistic Data Structures

The current fingerprint dedup store is an exact-match dict that grows unboundedly. Replacing it with a time-windowed Count-Min Sketch or HyperLogLog structure (e.g. via `pybloom-live`) gives O(1) insert/query with bounded memory and configurable false-positive tolerance — far more appropriate for high-cardinality signal streams.

**Impact**: Sub-millisecond dedup at millions of signals/hour with constant memory footprint.

---

## 7. RBAC-Aware Tenant Isolation at the Service Layer

`_enforce` delegates to `evaluate_capability_rules` but does not enforce row-level tenancy beyond the key lookup. Adding an explicit `TenantContext` dataclass carrying caller identity, roles, and workspace-scoped permissions — validated on every write path — prevents privilege escalation between tenants sharing a process.

**Impact**: Defense-in-depth isolation; required for multi-tenant SaaS deployment.

---

## 8. Structured Alert Playbooks with Runbook Automation

Resolution currently captures a `resolution_reference` string. A first-class `AlertPlaybook` model — a DAG of steps with preconditions, assigned actors, and automated action hooks — would guide analysts through standardised response workflows and track step completion, evidence attachment, and sign-off, reducing response variance.

**Impact**: Consistent incident response quality; measurable MTTR reduction.

---

## 9. Streaming Aggregation with Tumbling Windows

`alert_throughput` counts all-time totals rather than true time-windowed metrics. Maintaining tumbling 1-minute / 5-minute / 1-hour counters updated on every mutation (via a lightweight ring buffer) enables real-time rate-of-change alerting ("alert storm detection") and accurate SLA burn-rate forecasting.

**Impact**: Enables proactive capacity and SLA breach prediction.

---

## 10. Webhook / Push Notification Dispatcher

`record_notification` records intent but does not dispatch. An async `dispatch_notification` method that routes to HTTP webhooks, PagerDuty, OpsGenie, Slack, or email via a provider adapter pattern — with retry, back-off, and delivery receipt tracking — completes the notification lifecycle.

**Impact**: Closes the gap between "notification recorded" and "analyst actually notified."

---

## 11. Full-Text and Faceted Search over Alerts

There is no query surface beyond `list_rules` / `list_signals` with basic equality filters. Integrating a PostgreSQL `tsvector` full-text index on `alert_reference` and `evidence_reference`, plus a faceted filter API accepting arbitrary field predicates, would let analysts find relevant alerts in seconds rather than scanning exports.

**Impact**: Reduces mean time to investigate (MTTI) by enabling rapid evidence discovery.

---

## 12. Immutable Audit Log with Cryptographic Integrity

`audit_events` is a plain mutable list. An append-only audit table with per-row HMAC signatures (keyed by tenant secret) and periodic Merkle-root checkpoints makes the audit trail tamper-evident and suitable for compliance-grade forensic investigations.

**Impact**: Evidence integrity for legal / regulatory defensibility.

---

## 13. Graph-Based Alert Causality Inference

`correlate_alerts` creates groups manually. An automated causality inference pass — treating signals as nodes in a directed acyclic graph, with edges derived from shared rule ancestry, temporal proximity, and entity overlap — would surface root-cause candidates without analyst intervention, collapsing N alerts into 1 incident ticket.

**Impact**: Reduces alert fatigue; directly lowers mean time to root cause (MTTRC).

---

## 14. Zero-Trust Signal Provenance Verification

Signals are accepted at face value. Attaching a cryptographic provenance chain — each signal carries a signature from the originating rule engine, verified against a tenant-scoped public key at ingestion — ensures that signals cannot be injected or replayed without valid credentials, closing a significant threat-actor attack surface.

**Impact**: Prevents signal injection attacks; zero-trust posture for the entire pipeline.

---

## 15. Observability via OpenTelemetry Traces and Metrics

There is no distributed tracing or metrics instrumentation. Wrapping key methods with OpenTelemetry spans (`otel-sdk`) and emitting counter/histogram metrics (alert creation rate, SLA breach rate, agent action latency) to a Prometheus endpoint gives SREs real-time visibility into service health and alert pipeline performance without bespoke dashboards.

**Impact**: Production-grade observability; enables SLO alerting on the alerting system itself.
