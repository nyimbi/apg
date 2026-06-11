# int_api — World-Class Improvement Roadmap

© 2025 Datacraft. All rights reserved. Author: Nyimbi Odero <nyimbi@gmail.com>

These 15 improvements are ranked by leverage × feasibility within the existing in-memory service
architecture. Each one has a concrete implementation path against `service.py` and the surrounding
modules.

---

## 1. Circuit Breaker per Upstream

**Gap**: the `circuit_breaker` policy type exists in `SUPPORTED_POLICY_TYPES` but there is no state
machine tracking open/half-open/closed transitions per upstream. High-latency or error-rate spikes
will pass through indefinitely.

**Fix**: add `_circuit_state: dict[str, dict]` tracking `(upstream_url, tenant) → {state, failure_count,
last_failure_at, half_open_at}`. Wire `record_usage` to increment failure counts and trip the breaker
at a configurable threshold. Expose `async get_circuit_state(upstream_url)` and
`async reset_circuit(upstream_url)`.

**Impact**: prevents cascade failures; fulfils the `circuit_breaker` policy contract already declared
in the capability.

---

## 2. Distributed Rate-Limit Counters (Redis-Backed)

**Gap**: rate-limit enforcement is declared in policies but `service.py` never actually increments or
checks counters. `RateLimitExceededError` is defined but never raised.

**Fix**: add `async check_rate_limit(consumer_id, api_id, endpoint_id)` that reads/increments a
sliding-window counter. Behind an abstract `RateLimitBackend` protocol so the in-memory stub remains
the default; production slots in a Redis adapter.

**Impact**: makes the rate-limit governance guarantee executable rather than decorative. Directly
addresses the `api_rate_limit_positive` rule.

---

## 3. API Key Rotation with Zero-Downtime Overlap

**Gap**: `issue_api_key` creates keys but there is no rotation workflow. Operators have no way to
cycle credentials without a hard cutover, creating a gap window where traffic breaks.

**Fix**: `async rotate_api_key(key_id, overlap_seconds)` — issue a new key, mark the old key
`rotating` with a `deactivate_after` timestamp, return both keys. A background `async
complete_rotation(key_id)` deactivates the old key after the overlap window.

**Impact**: zero-downtime secret rotation; satisfies PCI-DSS / SOC 2 key-lifecycle controls.

---

## 4. OpenAPI Spec Validation and Diff on Registration

**Gap**: `AMAPI.openapi_spec` is a JSONB column but `register_api` ignores it. Operators cannot
validate an incoming spec or detect breaking changes on re-registration.

**Fix**: `async validate_openapi_spec(spec: dict)` — check required fields (`openapi`, `info`,
`paths`), resolve `$ref`s, return structured errors. `async diff_openapi_spec(api_id, new_spec)`
returns added/removed/changed paths, making breaking-change detection explicit.

**Impact**: prevents silent contract drift; enables automated compatibility gates in CI/CD pipelines.

---

## 5. Canary Traffic Splitting with Automatic Promotion

**Gap**: `DeploymentStrategy.CANARY` is modelled in `models.py` but `deploy_api` only records a
status; no traffic split state machine exists.

**Fix**: `async shift_canary_traffic(deployment_id, target_percentage)` — advance the
`traffic_percentage` field by a step. `async promote_canary(deployment_id)` — flip to 100 % and
mark the baseline version retired. `async rollback_canary(deployment_id, reason)` — revert to 0 %
and mark deployment `rolled_back`.

**Impact**: enables safe progressive rollouts with measurable blast radius; fulfils the `blue_green`
and `canary` deployment strategy contracts.

---

## 6. Structured Audit Log with Immutable Append Semantics

**Gap**: `_audit_events` is a plain list; events can be mutated or dropped. There is no query
interface, pagination, or tamper-evidence.

**Fix**: replace the list with a `BoundedImmutableLog` that: (a) appends only (no in-place
mutation), (b) hashes each entry with a chained HMAC (each event carries the hash of the previous
one), (c) exposes `async query_audit(tenant_id, event_types, since, limit)` with cursor-based
pagination.

**Impact**: satisfies the `audl` composability requirement; enables external audit-trail consumers
without a full SIEM deployment.

---

## 7. Webhook Retry with Exponential Back-off and Dead-Letter Queue

**Gap**: `test_webhook` simulates a single delivery with a hard-coded 200 OK. There is no retry
mechanism, so transient failures silently discard events.

**Fix**: `async deliver_webhook(webhook_id, event_type, payload)` — attempt delivery, record
outcome. On failure, enqueue to `_webhook_dlq` with a `next_retry_at = now + 2^attempt * base_ms`.
`async flush_webhook_dlq(webhook_id)` — replay all ready retries. `async webhook_dlq(webhook_id)`
— return pending DLQ entries.

**Impact**: guarantees at-least-once delivery semantics; required for reliable event-driven
integration partners.

---

## 8. Per-Tenant Configuration Overrides at Runtime

**Gap**: `get_capability_contract` accepts `overrides` but `IntApiService` never exposes a way to
set or retrieve per-tenant runtime configuration. All tenants run with the same `DEFAULT_CONFIGURATION`.

**Fix**: `async set_tenant_config(tenant_id, section, overrides: dict)` — persist overrides into
`_tenant_configs[tenant_id]`. Plumb `_tenant_config(tenant_id)` into `_assert_rules` so rules
evaluate against the tenant-specific rate limit thresholds, plan lists, and agent scopes.

**Impact**: enables multi-tenant SaaS deployment where each customer has negotiated different plan
tiers and latency thresholds.

---

## 9. SLA Budget Tracking with Burn-Rate Alerts

**Gap**: `data_quality_report` computes a quality score but there is no SLA budget concept.
Operators cannot tell how much error budget remains before an SLO is breached.

**Fix**: `async sla_budget(api_id, slo_target: float, window_hours: int)` — compute `error_budget_remaining =
(1 - slo_target) * total_requests - actual_errors`, derive `burn_rate`, flag `budget_exhausted` when
negative, emit a `sla_budget_alert` audit event. Store SLO definitions per API in `_slo_configs`.

**Impact**: converts raw latency/error data into actionable budget burn signals; enables automated
freeze gates on production deployments.

---

## 10. Schema Registry Integration for Event-Driven APIs

**Gap**: `register_integration` stores a free-form `config` dict. There is no schema validation for
the event payloads flowing through event-driven integrations (webhooks, Bytewax streams).

**Fix**: `async register_event_schema(integration_id, event_type, schema: dict)` — store a JSON
Schema in `_event_schemas`. `async validate_event_payload(integration_id, event_type, payload)` —
validate against the registered schema, return structured validation errors.

**Impact**: catches payload contract violations before they corrupt downstream consumers; enables
schema evolution gating.

---

## 11. Cost Attribution per Consumer

**Gap**: `AMUsageRecord.cost` and `AMSubscription.price_per_request` are modelled but never
populated by the service. There is no billing materialization path.

**Fix**: `async compute_consumer_cost(consumer_id, period_start, period_end)` — aggregate usage
records for the period, apply the subscription's pricing model (`free`, `usage`, `subscription`),
return `{consumer_id, total_requests, billable_requests, unit_cost, total_cost, currency}`. Store
results in `_billing_records`.

**Impact**: makes billing first-class; downstream finance systems can consume the output without a
separate ETL.

---

## 12. Dependency Graph and Impact Analysis

**Gap**: there is no way to determine which consumers, deployments, and webhooks depend on a given
API before changing or retiring it.

**Fix**: `async api_dependency_graph(api_id)` — traverse `subscriptions`, `deployments`,
`api_keys`, and `webhooks` to build a dependency graph:
`{api_id, consumers: [...], active_deployments: [...], webhooks: [...], downstream_apis: [...]}`.
`async impact_analysis(api_id, proposed_change)` — annotate each dependency with its risk level.

**Impact**: prevents accidental breaking changes; required before canary promotion or version
retirement.

---

## 13. Versioned API Snapshots and Rollback

**Gap**: `AMAPI` has a `version` field but `service.py` has no snapshot mechanism. Rolling back to
a previous API definition requires out-of-band tooling.

**Fix**: `async snapshot_api(api_id, label)` — deep-copy the full API record (endpoints, policies)
into `_api_snapshots[api_id][label]`. `async restore_api_snapshot(api_id, label)` — overwrite
current state from the snapshot, emit `api_restored` audit event.

**Impact**: enables point-in-time recovery for API definitions; fulfils the `rollback_available`
contract already on `AMDeployment`.

---

## 14. Adaptive Rate-Limit Tuning from Usage Patterns

**Gap**: rate limits are set at registration time and never adjusted. Usage patterns shift over time,
leaving limits either too restrictive (throttling legitimate traffic) or too permissive (allowing
abuse).

**Fix**: `async recommend_rate_limit(api_id, lookback_hours: int)` — analyse the P95 and P99
request cadence from `usage_records`, compute a recommended limit as `P99_rpm * safety_factor`,
return a recommendation dict. `async apply_recommended_rate_limit(api_id, approved_by)` —
materialise the recommendation with an audit trail.

**Impact**: turns rate limiting from a one-time guess into a data-driven control loop.

---

## 15. Multi-Region Deployment Coordination

**Gap**: `deploy_api` targets a single environment with no concept of region. Multi-region
deployments require separate calls with no coordination primitive.

**Fix**: `async deploy_multi_region(api_id, regions: list[str], strategy, deployed_by, approved_by)` —
fan-out `deploy_api` calls concurrently with `asyncio.gather`, return a per-region result map.
`async regional_health_summary(api_id)` — aggregate health checks across all regions, flag
divergent deployment versions.

**Impact**: enables geographically distributed API deployments with a single control-plane
operation; prerequisite for active-active multi-region architectures.
