# MCHN — World-Class Improvement Roadmap

**Capability**: Multi-Channel Output (`mchn`)
**Version**: 1.0.0 → 2.0.0
**Date**: 2026-06-11
**Author**: Nyimbi Odero — Datacraft
**Copyright**: © 2025 Datacraft

---

## Summary

The 15 improvements below transform `mchn` from a solid in-memory proof-of-concept into a
production-grade, horizontally scalable, observability-first output platform.
Each improvement is graded by impact tier (I = infrastructure, P = product, O = observability).

---

## Improvement 1 — Async-First Service Core [I]

**Problem**: `create_channel`, `publish_template`, `deliver_batch`, and related write methods are
synchronous. All callers block the event loop when `mchn` is hosted inside an ASGI application.

**Solution**: Promote every public write method to `async`. Internal helpers that reach out to
the database store (`store.py`) or Bytewax adapters must `await` those calls. Keep sync thin
wrappers (`_sync_*`) only for CLI and migration tooling.

**Impact**: Eliminates GIL contention under high throughput; enables concurrent batch rendering
with `asyncio.gather`.

---

## Improvement 2 — PostgreSQL-Backed Persistent Store [I]

**Problem**: All state lives in Python dicts. A process restart wipes every channel, template,
policy, route, rendered output, batch, receipt, and audit event.

**Solution**: Replace in-memory dicts with async SQLAlchemy sessions against the existing
`database/store.py` layer. Map each domain model to a DB table using the `mchn_` prefix already
defined in `database/schema.sql`. Use `alembic` (already scaffolded) for schema migrations.

**Impact**: Survives restarts, enables multi-instance deployments, unlocks DB-level filtering and
pagination.

---

## Improvement 3 — Event Streaming via Bytewax [I]

**Problem**: `deliver_batch` validates the `event_stream` parameter but never actually emits to
Bytewax. Downstream consumers (audit sinks, analytics workers) receive no signals.

**Solution**: Implement a `BytewaxOutputSink` adapter that publishes a CloudEvent envelope for
every state transition: `channel_created`, `batch_queued`, `receipt_recorded`,
`output_archived`, etc. The adapter contract is already declared in `capability_contract.py`.

**Impact**: Decouples delivery telemetry from the synchronous request path; enables replay-based
audit reconstruction.

---

## Improvement 4 — Rate-Limiter Enforcement in Delivery Policy [P]

**Problem**: `DeliveryPolicy.throttle_per_minute` is stored but never enforced. There is no
sliding-window counter protecting providers from burst traffic.

**Solution**: Add a `RateLimiter` component (token-bucket or leaky-bucket) to `output_runtime.py`.
`deliver_batch` checks the limiter before queuing; excess batches receive status `rate_limited`
and are held for the next window. Expose limiter state via `channel_analytics`.

**Impact**: Prevents provider bans, protects tenant SLAs, and enables fair-queuing between
tenants.

---

## Improvement 5 — Webhook Inbound Receipt Ingestion [P]

**Problem**: Provider delivery receipts are injected manually via `record_receipt`. Real providers
(SendGrid, Twilio, Firebase) push events to a webhook endpoint. There is no inbound handler.

**Solution**: Add `async def ingest_webhook(tenant_id, provider_id, payload)` that maps provider-
specific webhook shapes to `DeliveryReceipt` objects. Include an HMAC signature verifier per
provider. Emit `receipt_recorded` to Bytewax on success.

**Impact**: Enables real-time delivery confirmation without polling; prerequisite for SLA dashboards.

---

## Improvement 6 — Multi-Locale Template Variants [P]

**Problem**: Each `OutputTemplate` carries a single `locale`. Multi-language campaigns require
publishing one template per locale. There is no variant-linking mechanism.

**Solution**: Add a `template_variants` lookup keyed by `(template_id, locale)`. Add
`async def publish_template_variant(template_id, locale, subject_template, body_template,
approved_by)`. `render_output` resolves the best locale variant by BCP-47 fallback chain.

**Impact**: Reduces template proliferation, enables A/B locale testing, aligns with `i18n`
capability contract.

---

## Improvement 7 — A/B Test Routing [P]

**Problem**: There is no mechanism to split traffic between two routes or templates to measure
engagement outcomes. Experimentation requires external orchestration.

**Solution**: Add `async def create_ab_test(test_id, tenant_id, control_route_id, variant_route_id,
split_fraction, actor)` and integrate the split into `render_output` via a deterministic hash of
`recipient_ref`. Record which arm each recipient received. Expose results via `ab_test_results`.

**Impact**: Enables data-driven template and channel optimization without external tooling.

---

## Improvement 8 — Delivery SLA Tracking [O]

**Problem**: There is no tracking of time-to-delivery relative to a policy-defined SLA window.
SLA breaches are invisible until a manual audit.

**Solution**: Add `sla_window_minutes` to `DeliveryPolicy`. Record `queued_at` on `DeliveryBatch`
and `delivered_at` on `DeliveryReceipt`. Add `async def sla_report(tenant_id, batch_id)` that
returns per-recipient SLA compliance, breach count, and breach fraction.

**Impact**: Enables proactive alerting, provider SLA accountability, and compliance evidence
generation.

---

## Improvement 9 — Content Security Scanning [P]

**Problem**: Templates are published with approval gating but body content is never scanned for
PII leakage, injection patterns, or prohibited terms. Compliance reviews are manual.

**Solution**: Add `async def scan_template(template_id, tenant_id)` that runs a lightweight
rule-based scanner (regex + keyword list) and attaches a `scan_result` to the template record.
`render_output` blocks rendering if the scan result is `failed`. Scanner rules are configurable
via `conf`.

**Impact**: Prevents accidental PII exposure, satisfies GDPR Art. 25 data-protection-by-design
requirements.

---

## Improvement 10 — Idempotency Keys on Write Operations [I]

**Problem**: Network retries can create duplicate channels, templates, or batches. There is no
idempotency layer; the caller must track state externally.

**Solution**: Accept an optional `idempotency_key` on all create/publish/deliver write methods.
Hash the key and store the first response. Subsequent calls with the same key return the cached
response without side effects. Keys expire after a configurable TTL.

**Impact**: Safe retries; eliminates duplicate delivery on transient network failures.

---

## Improvement 11 — Structured Observability with OpenTelemetry [O]

**Problem**: The audit log (`MchnAuditEvent`) is internal and opaque. Operators have no spans,
metrics, or logs flowing to their observability stack.

**Solution**: Instrument every public service method with `opentelemetry-api` spans. Export
`mchn.delivery.sent`, `mchn.delivery.failed`, `mchn.channel.health`, and `mchn.batch.size`
metrics via OTLP. Use a no-op provider when OTEL is not configured.

**Impact**: Native Grafana/Datadog/Honeycomb visibility; on-call engineers can trace a delivery
failure end-to-end in under 30 seconds.

---

## Improvement 12 — Channel Circuit Breaker [I]

**Problem**: `selected_channel_id` falls back to the primary channel even when it is `unhealthy`.
Sending to a known-bad provider wastes quota and extends delivery failures.

**Solution**: Implement a half-open circuit breaker per `(tenant_id, channel_id)`. After N
consecutive failures the circuit opens and the channel is automatically marked `unhealthy`.
After a cool-down the circuit moves to half-open for a probe attempt. Expose breaker state in
`channel_health`.

**Impact**: Automatic recovery from transient provider outages; reduces wasted quota by 80–95%
during outages.

---

## Improvement 13 — Rendered Output Diff and Version History [P]

**Problem**: When a template is updated and re-rendered, there is no way to compare the new
output against the previous version. Regression detection requires manual inspection.

**Solution**: Add `async def output_diff(tenant_id, output_id_a, output_id_b)` that returns a
structured diff of `subject` and `body` fields using a Myers-diff algorithm. Store rendered output
versions keyed by `(tenant_id, route_id, recipient_ref, version)`.

**Impact**: Enables pre-send regression checks, template change audits, and rollback support.

---

## Improvement 14 — Bulk Suppression Import [P]

**Problem**: `suppression_add` accepts a list but has no bulk import path (CSV, JSON Lines) and
no deduplication guarantee across calls. Large unsubscribe lists require many API roundtrips.

**Solution**: Add `async def suppression_import(tenant_id, channel_type, source_uri, format,
actor)` that streams a file from a URI (local path or S3), deduplicates in a single pass, and
atomically merges into the suppression set. Return a summary with added/skipped/duplicate counts.

**Impact**: Reduces import time for million-row suppression lists from hours to minutes; prevents
duplicate suppression entries.

---

## Improvement 15 — Delivery Cost Budget Enforcement [P]

**Problem**: `channel_cost_report` estimates costs after the fact. There is no pre-delivery budget
gate. A misconfigured batch can exhaust a tenant's monthly budget in a single call.

**Solution**: Add `monthly_budget_usd` and `cost_per_message` to `DeliveryPolicy`. Before
`deliver_batch` queues a batch, it checks cumulative month-to-date spend against the budget.
If the batch would exceed the budget, delivery is blocked with `budget_exceeded` unless an
override is provided by an authorized actor. Expose spend vs. budget in `dashboard_summary`.

**Impact**: Prevents runaway spend; enables self-service cost governance without infrastructure
changes.

---

*Generated by Datacraft APG capability enhancement pipeline.*
