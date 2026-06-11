# Order Management — World-Class Improvement Catalogue

**Capability**: `telecom_ord` | **Domain**: Telecom | **Date**: 2026-06-11

---

## 1. Idempotent Order Submission

**Problem**: `submit_order` silently overwrites if called twice with the same `order_id`; duplicate detection only works via a loose heuristic on customer+status count.

**Improvement**: Store an idempotency key per `(tenant_id, order_id)` and return the original response on repeat calls within a configurable TTL (default 24 h). Eliminates ghost duplicates from retrying HTTP clients and makes the API safe to replay.

---

## 2. Finite-State-Machine Order Status Transitions

**Problem**: Status fields (`order.status`) are mutated directly with no guard on invalid transitions, e.g. moving `completed → submitted`.

**Improvement**: Declare an explicit FSM transition table and raise `InvalidTransitionError` on illegal moves. Enables formal correctness proofs and cleaner integration with workflow engines (`wflo`). Add `get_valid_transitions(order_id)` introspection endpoint.

---

## 3. Priority-Aware SLA Thresholds

**Problem**: `order_sla_monitoring` uses a single `sla_hours` float for all orders regardless of priority.

**Improvement**: Look up per-priority SLA from config: `emergency=1h`, `urgent=2h`, `high=4h`, `normal=24h`, `low=72h`. Breach and at-risk bands are computed independently per order. Surfaced in dashboards and exported to a Prometheus-compatible metrics endpoint.

---

## 4. Task Dependency Graph Execution

**Problem**: `depends_on` is stored as a raw string; there is no scheduler that honours actual task dependencies during execution.

**Improvement**: Build a DAG from `OrdTask.depends_on` references, topologically sort, and schedule parallel-safe groups concurrently with `asyncio.gather`. Expose `get_task_execution_plan(order_id)` returning the computed DAG as a JSON adjacency list.

---

## 5. Structured Fallout Root-Cause Taxonomy

**Problem**: Fallout categories are a flat enumeration; error codes map to categories via a hardcoded dict inside `order_fallout`.

**Improvement**: Introduce a `FalloutTaxonomy` registry (configurable per tenant) mapping `(error_code_prefix, system_domain)` → `(category, auto_retry_eligible, escalation_sla_minutes)`. Enables smarter retry decisions and feeds automated root-cause reporting.

---

## 6. Event-Sourced Audit Trail with Replay

**Problem**: Audit events are appended to an in-memory list with no structure beyond `event_type`/`reference_id`; there is no way to reconstruct order state from events alone.

**Improvement**: Emit structured `CloudEvent`-compliant records (via `situ-cloudevents`) for every state change. Add `replay_order(order_id, as_of_datetime)` that rebuilds order state from the event log up to a given timestamp — essential for dispute resolution and regulatory replay.

---

## 7. Number Portability Regulatory Compliance Checks

**Problem**: `submit_portability_request` stores the request but performs no regulatory pre-validation (MSISDN format, donor/recipient operator codes against a registry, porting window restrictions).

**Improvement**: Add `validate_portability_eligibility(msisdn, donor, recipient)` that checks MSISDN format (E.164), operator code validity against a configurable registry, and active porting window (no concurrent port-in/port-out for the same MSISDN). Returns structured eligibility report before committing the request.

---

## 8. Async Provisioning Webhook Callbacks

**Problem**: The service has no outbound notification mechanism; external systems poll for order state changes.

**Improvement**: Add `register_webhook(order_id, callback_url, events, secret)` that registers HMAC-signed webhook delivery for specific lifecycle events. Deliver via background task using `httpx.AsyncClient` with exponential back-off. Aligns with TMF622 Order Management API notification contract.

---

## 9. Multi-Tenant Data Isolation Enforcement

**Problem**: Dict lookups use `(tenant_id, item_id)` tuples correctly, but several list-based stores (`_amendments`, `_cancellations`, `_sla_events`) are not partitioned by tenant.

**Improvement**: Replace unpartitioned lists with `dict[str, list[...]]` keyed by `tenant_id`. Add `assert_tenant_isolation(tenant_id)` helper that validates every store access. Write a fuzz test that submits overlapping operations from two tenants and asserts zero data leakage.

---

## 10. Bulk Order Progress Streaming

**Problem**: `bulk_order_import` and `submit_bulk_order` are fire-and-forget; callers have no way to track progress on large batches.

**Improvement**: Introduce `stream_bulk_order_progress(bulk_id, tenant_id)` as an async generator that yields `{"processed": n, "total": N, "errors": [...]}` snapshots. Front-end can SSE-subscribe to the generator. Persists partial progress so restarts resume rather than replay from zero.

---

## 11. Contract Lifecycle Management

**Problem**: `contract_creation` stores a contract dict but there is no signing confirmation, renewal, or expiry handling.

**Improvement**: Add `confirm_contract_signature(contract_id, signed_by, signature_hash)`, `renew_contract(contract_id, extension_months)`, and `expire_contracts()` (cron-callable) that moves past-end-date contracts to `expired` status and fires renewal notifications via `ntfy`. Aligns with TMF651 Agreement Management API.

---

## 12. Order Jeopardy Prediction (ML-Assisted)

**Problem**: Jeopardy detection in `order_sla_monitoring` is purely time-based; no early signal from task completion patterns.

**Improvement**: Add `predict_order_jeopardy(order_id)` that scores risk using a lightweight feature set: age ratio (age/sla), fallout count, retry count, task completion rate, priority weight. Returns `{"risk_score": 0.83, "risk_band": "high", "recommended_action": "escalate"}`. Model is a configurable scoring function so it can be replaced with a trained classifier.

---

## 13. Order Cost Estimation

**Problem**: No pre-order cost visibility; customers discover pricing only after provisioning completes.

**Improvement**: Add `estimate_order_cost(customer_id, products, duration_months, tenant_id)` that computes itemised cost breakdown from a product catalogue (fetched from `telecom_inv`): `{product_id: {unit_price, quantity, subtotal}, "total": ..., "currency": "KES"}`. Integrates with `telecom_bil` for real-time tariff lookup.

---

## 14. Concurrent Order Deduplication Lock

**Problem**: Race condition: two concurrent `submit_order` calls with the same `order_id` both pass the duplicate check because neither has committed yet.

**Improvement**: Introduce an async `asyncio.Lock` keyed by `(tenant_id, order_id)` using a `defaultdict(asyncio.Lock)`. Both paths acquire the lock before mutating state. In distributed deployments, the lock is backed by Redis via `aioredis` with a short TTL, falling back to in-process lock for single-node setups.

---

## 15. Observability: Structured Metrics Export

**Problem**: `get_kpis` returns empty stubs; `order_analytics` computes on-demand but produces no time-series data.

**Improvement**: Instrument every public method with `time.perf_counter` latency recording stored in a ring buffer. Add `export_metrics(format="prometheus")` that emits `telecom_ord_method_duration_seconds{method,tenant}` histograms and `telecom_ord_order_total{status,channel,priority,tenant}` counters in Prometheus text format. Optionally push to OTLP endpoint if configured.
