# CONN World-Class Improvements

15 high-impact improvements to elevate the APG connectivity capability.

---

## 1. Circuit Breaker Pattern for API Connectors

**Problem:** Cascading failures when upstream APIs degrade. A single slow endpoint blocks the entire pipeline and exhausts worker threads.

**Solution:** Implement a per-connection circuit breaker with three states (closed/open/half-open), configurable failure thresholds, and exponential backoff. When the circuit opens, callers receive a fast `CircuitOpenError` instead of waiting for timeouts.

**Impact:** Eliminates cascading failure scenarios; reduces mean-time-to-recovery from minutes to seconds.

---

## 2. Adaptive Rate Limiting with Token Bucket

**Problem:** Static rate limits are either too conservative (leaving capacity unused) or too aggressive (triggering 429s and backoffs that stall pipelines).

**Solution:** Token bucket rate limiter that tracks per-connector quota windows. Automatically adjusts refill rate based on observed 429 response frequency. Supports per-tenant and per-connector limits.

**Impact:** Maximizes API throughput while staying within vendor-imposed rate limits; prevents quota exhaustion.

---

## 3. Change Data Capture (CDC) Native Support

**Problem:** Full-table polling is expensive and misses deletes. Singer taps default to full-refresh or simplistic bookmarks that don't capture row-level deletes.

**Solution:** Native CDC support via PostgreSQL logical replication slots (pgoutput), MySQL binlog consumer, and MongoDB change streams. Emit Singer-compatible RECORD/DELETE messages from the CDC feed so downstream targets receive true delta snapshots.

**Impact:** Reduces sync cost by 80-95% for large tables; enables sub-minute latency for database-sourced flows.

---

## 4. Dead Letter Queue (DLQ) with Automatic Replay

**Problem:** Records that fail transformation or target ingestion are silently dropped, causing invisible data loss.

**Solution:** Every failed record is written to a tenant-scoped DLQ with the original payload, error classification, and retry metadata. A background replay worker re-attempts DLQ records with exponential backoff. Operators can inspect, mutate, and re-enqueue DLQ records via API.

**Impact:** Zero-loss guarantee for transient failures; full observability into persistent errors.

---

## 5. Webhook Ingestion Engine with Signature Verification

**Problem:** Webhook endpoints are manually scaffolded per integration, lack signature verification, have no replay capability, and lose events during downtime.

**Solution:** Unified webhook ingestion engine. Verifies HMAC-SHA256/RSA signatures for all major providers (Stripe, GitHub, Shopify, HubSpot). Buffers events in a durable write-ahead log, deduplicates by event ID, and emits standardised Singer RECORD messages downstream.

**Impact:** Handles burst traffic without drops; prevents replay attacks; unifies 50+ webhook formats into one ingestion path.

---

## 6. Schema Registry with Backward/Forward Compatibility Enforcement

**Problem:** Schema drift silently corrupts downstream consumers. The current `detect_schema_drift` method flags changes but does not enforce compatibility or version schemas.

**Solution:** Embedded schema registry (Avro-compatible) with per-stream schema versions. Enforces backward/forward/full compatibility on schema evolution. Blocks registration of breaking changes unless explicitly overridden with a review token.

**Impact:** Prevents silent data corruption; gives operators a versioned schema changelog with rollback capability.

---

## 7. Connection Pool Manager with Adaptive Sizing

**Problem:** Each connection opens a new database/HTTP session. Under concurrent flow execution this causes connection exhaustion on the target system.

**Solution:** Per-connector async connection pool with adaptive sizing. Tracks utilisation and resizes between configured min/max bounds. Idle connections are periodically probed to prevent stale eviction. Pool metrics feed into the health monitor.

**Impact:** 3-5x throughput improvement under concurrent load; prevents `too many connections` errors.

---

## 8. End-to-End Data Lineage Graph

**Problem:** Current lineage tracking records individual events but does not materialise a queryable graph connecting source fields to target fields across multiple hops.

**Solution:** Build a directed acyclic graph (DAG) of field-level lineage. Each transformation rule creates an edge. The graph is queryable: "which source fields contribute to `warehouse.revenue`?" or "what downstream tables are affected if `orders.amount` changes type?".

**Impact:** Enables impact analysis before schema changes; satisfies GDPR right-to-erasure by identifying all tables holding a specific PII field.

---

## 9. Pluggable Secret Backend with Automatic Rotation

**Problem:** Credentials are stored in `tap_config` dictionaries. No rotation, no vault integration, and no audit trail when credentials change.

**Solution:** Abstract `SecretBackend` interface with drivers for HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager, and local encrypted file. Credentials are referenced by URI (`keym://tenant/secret-id`). Automatic rotation triggers reconnection tests and emits audit events.

**Impact:** Credentials never appear in config dicts or logs; rotation is automated and auditable.

---

## 10. Backpressure-Aware Streaming Pipeline

**Problem:** The current flow executor processes records in unbounded batches. If the target is slower than the source, memory pressure builds until the process OOMs.

**Solution:** Introduce an `asyncio.Queue`-backed pipeline with configurable watermarks. When the queue exceeds the high-water mark, the tap reader pauses. When it drops below the low-water mark, reading resumes. Each stage emits backpressure metrics.

**Impact:** Stable memory footprint regardless of source/target throughput asymmetry; enables safe processing of unbounded streams.

---

## 11. Multi-Tenant Credential Isolation with Row-Level Security

**Problem:** `tenant_id` is filtered in application code. A bug in a query predicate could expose one tenant's connections to another.

**Solution:** Enforce tenant isolation at the database level via PostgreSQL Row-Level Security (RLS) policies. Each DB session sets `app.current_tenant` and the RLS policy restricts all reads/writes to matching rows. Application-level filters remain as a defence-in-depth layer.

**Impact:** Cryptographic isolation between tenants; single-line application bug cannot cause cross-tenant data leak.

---

## 12. Observability Export (OpenTelemetry)

**Problem:** Metrics, traces, and logs are emitted via `print()`. No structured trace spans, no metric export, and no correlation between HTTP requests and Singer tap subprocess runs.

**Solution:** Instrument `ConnectionManager`, `FlowExecutor`, and `TransformationEngine` with OpenTelemetry spans. Export traces to OTLP-compatible backends (Jaeger, Grafana Tempo). Export `conn_*` metrics via Prometheus endpoint. Correlate HTTP request IDs into tap subprocess environments.

**Impact:** Full distributed trace from API request through tap execution to target write; P99 latency dashboards out of the box.

---

## 13. Idempotent Exactly-Once Delivery Semantics

**Problem:** Network retries cause duplicate records in targets. Current bookmarking only prevents re-reading from the source; it does not prevent double-writes to the target.

**Solution:** Each Singer RECORD message carries a deterministic `_apg_record_id` (hash of source PK + stream + state checkpoint). Targets de-duplicate on this ID using an upsert strategy. The bookmark is only advanced after the target confirms the write.

**Impact:** Exactly-once semantics end-to-end; safe for financial and audit-grade workloads.

---

## 14. AI-Assisted Connector Discovery and Auto-Configuration

**Problem:** Users must manually look up Singer tap names, read documentation, and write `tap_config` JSON. This is error-prone and requires deep knowledge of each connector.

**Solution:** Use the local Ollama LLM to accept natural-language intent ("connect to my Shopify store and sync orders to PostgreSQL") and produce a validated `tap_config` + `target_config` pair, suggest the correct Singer taps, and pre-fill sensible defaults. The AI suggestion is shown to the user with a confidence score before any connection is created.

**Impact:** Reduces connector setup time from hours to minutes; democratises access to non-technical operators.

---

## 15. Policy-as-Code Guardrails with OPA Integration

**Problem:** Governance rules (PII blocking, environment promotion, rate limits, required reviews) are scattered across Python conditionals. Adding a new rule requires code changes and a deployment.

**Solution:** Integrate Open Policy Agent (OPA) as the decision engine for all CONN guardrail evaluations. Policies are Rego documents stored in a versioned policy store. The `capability_contract.py` rule set is migrated to Rego. Policy changes are evaluated, tested, and deployed without application code changes.

**Impact:** Governance rules are auditable, version-controlled, and testable independently of application code; enables compliance teams to own policy without engineering involvement.
