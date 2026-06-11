# TENS Capability — World Class Improvements

**Capability**: Tenants Legacy (`tens`)
**Domain**: `common`
**Date**: 2026-06-11

---

## 1. Async-First Service Layer

All mutating and I/O-bound operations remain synchronous today. Promote them to `async def` with `asyncio`-compatible signatures so the service slots cleanly into async APG composition pipelines (FastAPI, Starlette, async Bytewax processors). Blocking helpers stay sync; public methods become awaitable.

## 2. Persistent Storage via SQLAlchemy Async ORM

Replace the in-memory `dict` stores with `sqlalchemy.ext.asyncio` sessions backed by PostgreSQL. Use Alembic migrations (skeleton already present) to version schema changes. In-memory dicts become a valid ephemeral backend only for unit tests, swapped in via a `StorageBackend` protocol.

## 3. StorageBackend Protocol / Adapter Pattern

Introduce a `StorageBackend` protocol with `get`, `put`, `delete`, `list`, and `query` methods. Ship two implementations: `InMemoryBackend` (current behaviour) and `PostgresBackend`. This decouples service logic from persistence and enables drop-in test doubles without mocks.

## 4. Structured Event Publishing via CloudEvents + Bytewax

Today, `_record_event` appends to an in-memory dict and returns raw JSON blobs. Replace with a typed `TensEvent` CloudEvents envelope and publish to a configurable `EventSink` (in-memory queue, Bytewax topic, or Redis Streams). Each event should carry `specversion`, `type`, `source`, `subject`, `datacontenttype`, and `data` fields per the CloudEvents 1.0 spec.

## 5. Pydantic v2 Input/Output Contracts on Every Public Method

Wrap every public method's inputs in a typed `Request` Pydantic model and every return value in a typed `Response` model. Eliminates `dict[str, Any]` surface area, gives runtime validation for free, and produces OpenAPI schemas automatically. Use `model_config = ConfigDict(extra='forbid', validate_by_name=True)`.

## 6. Deterministic Idempotency Keys

Operations that create records (register, map, plan, etc.) accept an optional `idempotency_key: str`. On replay, the service returns the stored result rather than raising or creating duplicates. Backed by a Redis-or-Postgres keyed store with configurable TTL.

## 7. Tenant Lifecycle State Machine

Replace ad-hoc `legacy.status = "..."` assignments with an explicit finite state machine (transitions: `active → stale → mapped → migration_ready → migrated → deprecated → archived → restored → suspended → merged`). Enforce valid transitions; raise `InvalidTransitionError` with the current/attempted states on illegal moves.

## 8. Bulk / Batch Operations with Partial Failure Reporting

Add batch variants for `register_legacy_tenant`, `map_tenant`, and `validate_access_boundary`. Each batch method returns a `BatchResult` with `succeeded: list`, `failed: list[FailureDetail]`, and `summary` counters. Partial success is acceptable; caller decides whether to roll back.

## 9. Optimistic Locking / ETag Concurrency Control

Add a `version: int` field to every persistent record. Mutating methods accept an optional `expected_version: int`. If the stored version differs, raise `ConcurrentModificationError`. Clients use the returned `version` as their ETag for subsequent updates.

## 10. Tenant Compliance Report

Add `async compliance_report(tenant_id, framework="SOC2")` that evaluates a tenant against a configurable compliance checklist (boundary present, migration plan approved, no stale tenants over threshold, audit log coverage ≥ N days). Returns a structured `ComplianceReport` with per-control pass/fail and an overall posture score.

## 11. Cross-Tenant Dependency Graph

Add `async dependency_graph(tenant_id)` that traverses mappings, boundaries, migrations, and merge records to produce an adjacency list and a topological sort. Useful for migration sequencing — surfaces which tenants must migrate before others.

## 12. Tenant Activity Scoring Model

Replace the binary `days_since_activity > 90 → stale` heuristic with a configurable `ActivityScorer` that weights API calls, event frequency, mapping age, and boundary recency into a 0–100 score. `health_check` consumes the score; thresholds are tenant-scoped configuration values.

## 13. Audit Log Integrity with HMAC Chaining

Chain audit events with HMAC-SHA256: each event stores `prev_hash` = HMAC of the previous event's content. `verify_audit_chain(tenant_id)` recomputes the chain and returns the first broken link if tampered. Makes the audit log tamper-evident without a separate ledger service.

## 14. Observability: Structured Logging + OpenTelemetry Spans

Wrap every public method in an `@otel_span("tens.<method>")` decorator that emits a span with `tenant_id`, `operation`, and result status as span attributes. Emit structured log lines (JSON) at DEBUG/INFO/WARNING levels with the same correlation fields. Zero-cost when a tracer is not configured.

## 15. Rate Limiting and Quota Enforcement Middleware

Integrate the existing `resource_quota` records into a `QuotaEnforcer` that checks `max_api_calls` per tenant per rolling window before every mutating operation. If quota is exceeded, raise `QuotaExceededError` with `limit`, `used`, and `reset_at`. The enforcer is a pluggable component; default implementation uses in-memory sliding windows.
