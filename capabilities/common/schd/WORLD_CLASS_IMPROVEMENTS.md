# SCHD — World-Class Improvements

Capability: Scheduler (schd) | Path: capabilities/common/schd
© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Async-Native Service Layer

The entire `SchdService` is synchronous. Every blocking operation (DB I/O, event
emission, SLA checks, heartbeat polls) will stall an event loop when the service
is embedded in an async application. Introduce an `AsyncSchdService` subclass
that wraps all public methods with `async def` and delegates DB calls to an
async adapter via `asyncio.to_thread` or native async drivers (asyncpg,
SQLAlchemy 2 async). Result: zero contention in high-throughput Bytewax pipelines.

## 2. Persistent PostgreSQL Backend via SQLAlchemy 2 Async

The in-memory dicts are dev scaffolding only. Replace them with a proper
`AsyncSession`-backed store using SQLAlchemy 2 `mapped_column` / `MappedAsDataclass`
models that mirror the existing dataclasses. Alembic migrations already scaffolded
in `alembic/versions/0001_initial.py`; finish them and wire `AsyncSchdStore` to
the service. Zero API change; only the data layer swaps.

## 3. Cron Expression Parser with Next-N-Runs Preview

`cron_validate` uses a naive regex. Replace with `croniter` (or a vendored
equivalent) to: (a) parse standard 5/6-field expressions, (b) compute the next N
run timestamps from any reference time, (c) detect always-false expressions
(`0 0 30 2 *`), and (d) return a human-readable English description. Surfaces
directly in the schedule-creation UI as a preview panel.

## 4. Distributed Lock / Idempotency Guard

Nothing prevents two concurrent callers from triggering the same schedule
simultaneously. Add `async def acquire_run_lock(schedule_id, ttl_seconds)` backed
by Redis SETNX (or a Postgres advisory lock). `trigger_run` calls it before
creating a `JobRun`; duplicate triggers within the TTL receive the existing run ID
instead of spawning a second run. Idempotency key stored alongside each run for
safe replay.

## 5. Backpressure-Aware Dispatcher

`trigger_run` enqueues work without checking whether the worker pool is at
capacity. Add `async def dispatch_with_backpressure(tenant_id, schedule_id, ...)`
that reads `queue_depth` against `max_concurrency` and either (a) enqueues
immediately, (b) parks the run in a `queued` state for later dispatch, or (c)
rejects with `worker_pool_at_capacity`. Pair with `async def flush_queued_runs`
for the scheduler loop to drain parked work.

## 6. Tenant-Aware Rate Limiting

No per-tenant throttle exists. Add `async def check_rate_limit(tenant_id, op)`
backed by a token-bucket in Redis (or in-memory `asyncio` deque for dev mode).
Per-tenant limits configured in `DEFAULT_CONFIGURATION`. Exceeding the limit raises
`RateLimitError` with a `retry_after_seconds` field. Protects shared infrastructure
in multi-tenant deployments.

## 7. Dependency-Graph Execution Engine

`dependency_chain` only records the relationship. Build `async def run_dag(tenant_id,
root_schedule_id, actor)` that topologically sorts the registered chain and fires
each dependent schedule only when its upstream runs complete successfully. Track DAG
run state in a `DagRun` model. Unblocks complex ETL pipelines currently wired with
fragile cron offsets.

## 8. SLA Alerting with Webhook / Notification Fanout

`sla_monitor` returns a report but never fires an alert. Extend it with
`async def sla_alert(tenant_id, schedule_id, channels)` that evaluates breach
conditions and calls registered webhook endpoints (Slack, PagerDuty, custom)
via `httpx.AsyncClient`. Retry logic reuses the existing `backoff_seconds`
implementation. Alert state stored per run so repeat notifications are suppressed.

## 9. Schedule Forecast & Capacity Planning API

Add `async def forecast_schedule_load(tenant_id, horizon_hours, resolution_minutes)`
that iterates over all active cron/interval schedules, projects their next-N runs
within the horizon, and aggregates expected concurrency per worker pool per time
bucket. Returns a time-series dict suitable for charting. Enables proactive
scale-out decisions before business-critical windows.

## 10. Run Metrics Streaming to Bytewax

Each `complete_run` call computes metrics in-process but never streams them.
Add `async def emit_run_metrics(run, job)` that publishes a CloudEvents-shaped
payload to the Bytewax topic configured in `DEFAULT_CONFIGURATION["streaming"]`.
Include: `run_duration_ms`, `records_processed`, `error_rate`, `sla_breached`,
`retry_attempt`. Downstream processors can build real-time dashboards without
polling.

## 11. Pluggable Calendar Holiday Provider

`CalendarPolicy.holiday_calendar` is a string reference with no resolution logic.
Define `HolidayProvider` protocol with `async def is_holiday(date, timezone) -> bool`.
Ship an `IcsHolidayProvider` (downloads iCal feed) and a `StaticHolidayProvider`
(JSON file). Inject into `SchdService.__init__`. `trigger_run` refuses to start on
blackout dates unless `force=True` is passed with an operator reason.

## 12. Structured JSON Logging with Correlation IDs

All `_record_event` calls write to an in-memory list. Replace with
`structlog` (or stdlib `logging` + `python-json-logger`) and attach a
`correlation_id` (UUID7) to every operation. Service methods accept an optional
`ctx: RequestContext` carrying `trace_id`, `span_id`, and `tenant_id`. Log lines
are machine-parseable and indexable by any log aggregator without schema changes.

## 13. OpenTelemetry Tracing Integration

Instrument every public service method with `opentelemetry-api` spans:
`tracer.start_as_current_span("schd.trigger_run")` captures duration, status,
and key attributes (`tenant_id`, `schedule_id`, `worker_pool_id`). No hard
dependency on a specific backend — exporter configured via env var. Adds zero
latency in production when no exporter is configured (no-op tracer).

## 14. Capability Health Probe Endpoint

Expose `async def health(tenant_id) -> HealthStatus` that checks: (a) DB
connectivity (ping), (b) Bytewax broker reachability, (c) worker pool heartbeat
freshness (stale if last heartbeat > 2x expected interval), and (d) dead-letter
queue depth against a configurable threshold. Aggregates into
`{"status": "healthy"|"degraded"|"unhealthy", "checks": [...]}`. Wired to
`/schd/health` in the Flask-AppBuilder blueprint for load-balancer probes.

## 15. Zero-Downtime Schedule Migration API

When a job's command, worker pool, or cron expression changes, operators
currently must disable the old schedule and create a new one — a race window
where runs can fire against stale config. Add `async def migrate_schedule(tenant_id,
schedule_id, patch: SchedulePatch, actor, dry_run=False)` that: (a) validates the
patch, (b) waits for any active run to complete (or cancels it with reason), (c)
atomically replaces the schedule definition, (d) recalculates `next_run_hint`,
and (e) emits a `schedule_migrated` audit event. `dry_run=True` returns a diff
without committing.
