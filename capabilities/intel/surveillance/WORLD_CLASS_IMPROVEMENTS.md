# World-Class Improvements — intel_surveillance

15 prioritised improvements to elevate `DigitalSurveillanceService` to production grade.

---

## 1. Persistent Storage via Async SQLAlchemy

All in-memory dicts (`authorities`, `programs`, `sensors`, …) must be backed by an async
SQLAlchemy `AsyncSession` (PostgreSQL). The injected `db_url` constructor argument is already
wired; implement `_async_session()` and replace every `self.<store>[key] = item` with a
`session.add(orm_item)` + `await session.commit()`. Queries become typed `select()` statements.
Removes entire class of data-loss on restart.

## 2. Authoritative Authority Expiry Enforcement

`register_surveillance_target` verifies `authority_ref` existence but not expiry. Parse
`expires_at` as `datetime` and reject registrations against expired authorities at all entry
points (`record_observation`, `location_tracking`, `lawful_intercept`, etc.). Add an
`async def expire_stale_authorities()` cron-callable that bulk-marks expired records.

## 3. Structured Event Streaming via Bytewax Producer

The `_audit` helper logs to a Python list. Replace with a real async Bytewax/Kafka producer
call:

```python
await self._stream_producer.send("apg.intel.surveillance.lifecycle", payload)
```

Inject `stream_producer: AsyncProducer | None = None` in `__init__` and fall back to the list
only when `None`. Enables real-time downstream consumers for alerting and compliance reporting.

## 4. Pydantic v2 Request/Response Models for Every Public Method

Every public method currently takes raw `str`/`float` positional args and returns bare
`dict[str, Any]`. Add a matching `SurveillanceXxxRequest` / `SurveillanceXxxResponse` Pydantic
v2 model pair (in `views.py`) for each workflow. Validates inputs before they reach the service
layer, generates OpenAPI schemas automatically, and removes all `assert present(...)` guards
in favour of Pydantic field validators.

## 5. RBAC Permission Check on Every Entry Point

`_enforce` calls `evaluate_capability_rules` but does not verify that `self.actor_id` holds the
required role. Inject an `auth` adapter with `async def check_permission(actor_id, action,
resource) -> bool`. Gate every write method behind `await self._auth.check_permission(...)`.
Provides fine-grained role separation between collectors, analysts, and reviewers.

## 6. Distributed Caching for `pattern_of_life` Computations

`pattern_of_life` recomputes from raw tracks on every call. Introduce a `BoundedCache`
(already imported from `capabilities.common.reliability`) keyed on `(tenant_id, target_id,
period)` with a 5-minute TTL. For production, swap the in-process cache for a Redis async
adapter via the injected `store` collaborator.

## 7. Physical Surveillance Coordination Module

The capability name implies physical surveillance but the service only handles digital signals.
Add `async def field_agent_tasking(target_id, agent_id, observation_zone, priority)` and
`async def observation_report_ingest(target_id, agent_id, report_text, media_refs)` to bridge
physical-world observation reports into the digital analytics pipeline.

## 8. Target Profile Builder

Add `async def build_target_profile(target_id) -> TargetProfile` that aggregates and
denormalises all available data (registration, location centroid, comm metadata, footprint
score, cross-platform correlations, pattern-of-life, associate network) into a single
structured `TargetProfile` Pydantic model. Acts as the single read-through entry point for
downstream consumers.

## 9. Time-Series Location History with Trajectory Analysis

`_location_tracks` stores a flat dict without ordering or trajectory computation. Replace with
a `list[LocationFix]` per target stored in insertion order. Add
`async def trajectory_analysis(target_id, window_hours) -> TrajectoryReport` that computes
speed, heading, mode-of-transport estimate (PEDESTRIAN / VEHICLE / STATIONARY), and flags
impossible jumps (>300 km/h between consecutive fixes).

## 10. Media Evidence Chain-of-Custody

Many methods accept `evidence_reference` as an opaque string. Add an evidence registry:
`async def register_evidence(evidence_id, reference_url, sha256_hash, custodian_id, expiry)`
and verify `sha256_hash` matches a re-computed hash at ingest. Every subsequent method that
consumes `evidence_reference` must resolve it against this registry, providing an unbroken
chain-of-custody for legal admissibility.

## 11. Automated Legal Authority Renewal Workflow

Before an authority expires, the system should surface a renewal task. Add
`async def check_authority_renewals(days_ahead: int = 30) -> list[AuthorityRenewalAlert]`
that scans all active authorities, identifies those expiring within `days_ahead`, and fires
notifications via the injected `notify` adapter. Prevents surveillance gaps caused by lapsed
legal cover.

## 12. Multi-Tenant Isolation Hardening

The `_tenant_key` tuple strategy works for in-memory maps but provides no isolation guarantee
at the query level. Add a tenant-scoping decorator `@tenant_scoped` that asserts
`self.tenant_id == item.tenant_id` for every retrieval, and a `cross_tenant_access_audit()`
method that scans audit events for cross-tenant reference IDs and raises `SecurityWarning` if
detected.

## 13. Sensor Calibration Scheduler

`sensor_health_check` reports overdue calibration but takes no corrective action. Add
`async def schedule_calibration(sensor_id, scheduled_at, technician_id)` and
`async def record_calibration_result(sensor_id, outcome, evidence_reference)` to close the
loop. Track calibration history per sensor and block `record_observation` calls from sensors
with overdue calibration.

## 14. Async Batch Observation Ingestion with Deduplication

`record_observation` is a synchronous method called one-at-a-time. For high-volume sensor
feeds, add `async def ingest_observations_batch(observations: list[ObservationIngestionItem])
-> BatchIngestionResult` that processes up to 500 observations concurrently, deduplicates
by `content_fingerprint` within the batch, and returns per-item success/skip/failure status
with a Bytewax stream emit for each successful ingest.

## 15. GraphQL or OpenAPI Auto-Generation from Service Signature

The `api.py` file currently contains manually authored Flask routes. Add a
`generate_openapi_spec() -> dict` class method that introspects public method signatures and
docstrings (via `inspect` + Pydantic schema extraction) to produce a standards-compliant
OpenAPI 3.1 document. This eliminates drift between implementation and documentation and feeds
API gateway configuration automatically.
