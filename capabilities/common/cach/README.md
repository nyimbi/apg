# CACH - Cache Management

CACH is APG's cache governance and runtime-adapter capability. It gives
generated applications a tenant-aware way to register cache namespaces, enforce
entry admission rules, manage warming and eviction reviews, publish UI metadata,
compose first-class cache agents, validate Bytewax lifecycle batches, and
connect to cache backends without hard-coding a specific store.

The current packet is intentionally dependency-light. It can publish a capability
contract, evaluate guardrails, build view models, and run focused tests without
Redis, Flask-AppBuilder, AI services, or optional compression packages.

## What CACH Provides

- Tenant-scoped namespace registration and policy defaults.
- Deterministic cache guardrails for writes, reads, warming, eviction, tenant
  isolation, encryption, TTL limits, and memory pressure.
- Entry lifecycle records for admitted, denied, expired, invalidated, and
  refresh-required cache entries.
- Warming plans with source evidence, batch limits, and review state.
- Eviction and capacity reviews with independent reviewer evidence.
- First-class cache-agent registration for Codex, Claude Code, opencode, and Pi
  with owner, purpose, scope, contribution disclosure, and human approval
  guardrails.
- Durable review evidence across cache entries, warming plans, eviction reviews,
  privileged cache agents, lifecycle batches, and audit events.
- Pending-review queue composition for generated cache operations consoles.
- Bytewax lifecycle-batch validation for cache policy, warming, agent, and
  eviction mutations.
- Compact generated-application view models for operations UIs.
- Theme tokens and component metadata for cache dashboards and control surfaces.
- Contract-derived semantic-model and release evidence for APG publish tooling.
- Adapter boundaries for memory, Redis-compatible stores, edge caches, CDNs,
  application-local query caches, and future APG integrations.

### New in v1.1 — World-Class Enhancements

- **Stale-While-Revalidate (SWR)**: `cache_set_swr` / `cache_get_swr` serve
  stale values immediately during the grace window while triggering async
  background revalidation via a registered refresh callback.
- **Adaptive TTL**: `adaptive_ttl_configure` + `cache_get_adaptive` extend TTL
  on each hit (up to `ttl_max`) and let cold entries expire naturally.
- **Schema-Versioned Writes**: `cache_set_versioned` / `cache_get_versioned`
  tag entries with a schema version; stale-schema hits auto-invalidate.
- **Cascading Tag Invalidation**: `register_tag_hierarchy` + `tag_invalidate_cascade`
  perform BFS traversal of the tag graph to invalidate all descendant tags in
  one call.
- **Streaming Cache Warm**: `warm_cache_stream` accepts any `AsyncIterator[tuple[key, value]]`,
  processes in configurable batches, and reports progress via callback —
  eliminating memory spikes on large warm-up datasets.
- **Monetary Value Cache**: `cache_set_money` / `cache_get_money` store
  `Decimal` amounts as strings to preserve precision; returned as `Decimal`.
- **XFetch Stampede Protection**: `xfetch_get` uses the Vattani et al. (2015)
  probabilistic early-expiry algorithm to distribute recompute load before
  expiry, eliminating thundering-herd failures.
- **Tenant Quota Enforcement**: `set_tenant_quota` + `quota_usage_report`
  enforce soft-warning and hard-limit byte quotas per tenant.
- **Tier Statistics**: `tier_stats` reports L1 (hot) vs L2 (warm/cold) entry
  distribution and overall hit rates per namespace.
- **Write-Behind Mode**: `cache_set_write_behind` + `write_behind_flush`
  decouple cache writes from backend persistence, reducing write latency to
  in-memory speeds.

## Key Files

- `SPECIFICATION.md` - full functional and guardrail specification.
- `PLAN.md` - implementation plan and deferred runtime work.
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 detailed improvement proposals with
  justification, implementation notes, and competitor references.
- `capability_contract.py` - configuration, rule engine, UI routes, and theme.
- `service.py` - 63-method async cache runtime: governance, SWR, adaptive TTL,
  schema versioning, cascading tag invalidation, streaming warm, monetary cache,
  XFetch anti-stampede, tenant quotas, tier stats, and write-behind.
- `api.py` - FastAPI routes plus direct helper functions for generated apps.
- `view_models.py` - generated-application UI model builders.
- `app.py` - APG package entrypoint and semantic model.
- `docs/user_guide.md` - comprehensive user guide with usage examples.
- `semantic_model.json` - publishable semantic-model evidence.
- `release_report.json` - focused release evidence.
- `tests/` - focused package and contract coverage.

## Runtime Shape

CACH has two layers:

1. **Control plane**: namespace policies, rule decisions, lifecycle records,
   review queues, audit events, UI metadata, and semantic evidence.
2. **Data plane adapters**: memory, Redis/Valkey, edge/CDN caches, query caches,
   AI optimizers, monitoring, audit, and security integrations.

Generated APG applications should call the control plane before touching a cache
backend. Cache adapters should honor the decisions returned by CACH.

## Direct Usage

```python
from capabilities.common.cach.api import (
    create_namespace_record,
    write_cache_entry_record,
    read_cache_entry_record,
    request_eviction_review,
    decide_eviction_review,
    register_cache_agent,
    validate_cache_lifecycle_batch,
)

namespace = create_namespace_record(
    tenant_id="tenant-a",
    namespace="orders",
    owner="platform",
    data_classification="regulated",
    max_ttl_seconds=900,
    encryption_required=True,
)

entry = write_cache_entry_record(
    tenant_id="tenant-a",
    namespace="orders",
    key="order:1001",
    value_ref="redis://orders/order:1001",
    producer="order-service",
    ttl_seconds=300,
    encrypted=True,
    data_classification="regulated",
)

read_result = read_cache_entry_record(
    tenant_id="tenant-a",
    namespace="orders",
    key="order:1001",
)

review = request_eviction_review(
    tenant_id="tenant-a",
    namespace="orders",
    requester="platform",
    memory_utilization_percent=94,
    proposed_action="evict cold entries from distributed tier",
    reason="tenant quota pressure",
)

decision = decide_eviction_review(
    review_id=review.review_id,
    reviewer="cache-sre",
    decision="approved",
    notes="Cold entries can be evicted; source of truth is healthy.",
)

agent = register_cache_agent(
    tenant_id="tenant-a",
    agent_id="warming-agent",
    name="Warming Agent",
    runtime="claude-code",
    role="warming-reviewer",
    scope="warming plan review",
    owner="platform",
    purpose="review warming plans",
    contribution_disclosed=True,
    human_approval_required=True,
)

batch = validate_cache_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=3,
)
```

Privileged cache agents that are otherwise valid but missing human approval are
stored as `pending_review` with `policy_decision="require_review"`. Denied
non-Bytewax lifecycle batches are stored as `denied` before `PermissionError` is
raised. Generated applications can use `list_pending_reviews()` or
`list_cache_governance()` to build a single review queue.

## Rule Evaluation

```python
from capabilities.common.cach.capability_contract import evaluate_capability_rules

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "write",
    "namespace_present": True,
    "namespace_status": "active",
    "data_classification": "sensitive",
    "entry_encrypted": False,
})

assert decision["decision"] == "deny"
assert "sensitive_entry_requires_encryption" in decision["matched_rules"]
```

## View Models

Generated applications can use `view_models.py` to render CACH without knowing
the internal service representation:

```python
from capabilities.common.cach.api import SERVICE
from capabilities.common.cach.view_models import dashboard_model, warming_console_model

dashboard = dashboard_model(SERVICE)
warming = warming_console_model(SERVICE, tenant_id="tenant-a")
```

## Permissions

The capability registration exposes these permissions:

- `cach:view`
- `cach:read`
- `cach:write`
- `cach:delete`
- `cach:manage_namespaces`
- `cach:manage_policies`
- `cach:warm`
- `cach:review_eviction`
- `cach:view_analytics`
- `cach:manage_agents`
- `cach:admin`

## Adapter Boundary

CACH does not require a specific backend. A production adapter should:

1. Ask CACH for a namespace and rule decision.
2. Store data in the selected backend only when the decision allows it.
3. Emit audit and telemetry events through APG adapters when available.
4. Preserve tenant isolation in backend key construction.
5. Respect encryption, TTL, invalidation, freshness, and eviction decisions.
6. Preserve CACH policy evidence fields when moving records between durable
   storage and runtime adapters.

## New Feature Usage Examples

### Stale-While-Revalidate

```python
from capabilities.common.cach.service import CacheService

svc = CacheService(actor_id="api", tenant_id="acme")

# Register a refresh callback
async def my_refresh(namespace, key):
    return await db.fetch(key)

await svc.register_refresh_callback("products", my_refresh)

# Write with a 60-second SWR grace window
await svc.cache_set_swr("products", "prod:42", {"name": "Widget"}, ttl_seconds=300, swr_grace_seconds=60)

# Read — stale=True triggers background revalidation within the grace window
result = await svc.cache_get_swr("products", "prod:42")
assert result["hit"] is True
# result["stale"] is True if entry was expired but within grace window
```

### Adaptive TTL

```python
await svc.adaptive_ttl_configure("session", ttl_min_seconds=120, ttl_max_seconds=7200, growth_factor=1.5)
await svc.cache_set("session", "user:99", {"role": "admin"}, ttl_seconds=300)

# TTL extends by 1.5x on each hit (up to 7200s)
result = await svc.cache_get_adaptive("session", "user:99")
# result["ttl_extended_to_seconds"] present when TTL was grown
```

### Schema-Versioned Cache

```python
await svc.cache_set_versioned("orders", "order:1001", {"id": 1001, "v": 2}, schema_version="2")

# Stale-schema entries auto-invalidate
r = await svc.cache_get_versioned("orders", "order:1001", expected_version="2")
assert r["version_mismatch"] is False

r_old = await svc.cache_get_versioned("orders", "order:1001", expected_version="1")
assert r_old["version_mismatch"] is True  # entry deleted, forces re-populate
```

### Cascading Tag Invalidation

```python
await svc.register_tag_hierarchy("catalog", "user:42", ["user:42:orders", "user:42:profile"])
await svc.cache_set("catalog", "k1", "v1", tags=["user:42:orders"])
await svc.cache_set("catalog", "k2", "v2", tags=["user:42:profile"])

result = await svc.tag_invalidate_cascade("catalog", "user:42")
# result["total_invalidated"] == 2 (both child-tag entries removed)
```

### Streaming Cache Warm

```python
async def db_cursor():
    async for row in db.stream("SELECT id, data FROM products"):
        yield (f"prod:{row.id}", row.data)

result = await svc.warm_cache_stream(
    "products", db_cursor(), ttl_seconds=3600, batch_size=200,
    progress_callback=lambda loaded, failed, ms: print(f"{loaded} loaded"),
)
print(result["loaded"], result["elapsed_ms"])
```

### Monetary Value Cache

```python
from decimal import Decimal

await svc.cache_set_money("pricing", "price:USD:prod42", Decimal("19.99"), "USD")

r = await svc.cache_get_money("pricing", "price:USD:prod42")
assert isinstance(r["amount"], Decimal)
assert r["amount"] == Decimal("19.99")
```

### XFetch Stampede Protection

```python
# beta=1.0 is standard; increase for more aggressive early recompute
r = await svc.xfetch_get("products", "prod:42", beta=1.0)
if not r["hit"]:
    # recompute and re-cache — safely distributed over time
    new_val = await source.fetch("prod:42")
    await svc.cache_set("products", "prod:42", new_val, ttl_seconds=600)
```

### Tenant Quota Enforcement

```python
await svc.set_tenant_quota("acme", soft_bytes=50_000_000, hard_bytes=100_000_000)
report = await svc.quota_usage_report("acme")
print(report["utilisation_pct"], "%")
```

### Write-Behind Mode

```python
# Fast write to cache, backend write deferred
await svc.cache_set_write_behind("orders", "order:9999", order_data, ttl_seconds=900)

# Drain the queue (call from a background task or scheduler)
result = await svc.write_behind_flush(backend_fn=async_db_writer)
print(result["flushed"], "writes persisted")
```

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/cach/capability_contract.py \
  capabilities/common/cach/service.py \
  capabilities/common/cach/api.py \
  capabilities/common/cach/view_models.py \
  capabilities/common/cach/app.py

./.venv/bin/pytest -q \
  capabilities/common/cach/tests/test_capability_contract.py \
  capabilities/common/cach/tests/test_package_contract.py

./.venv/bin/apg capabilities publish-plan capabilities/common/cach --json
```

Full repository tests, live cache backends, production persistence, APG
auth/audit/monitoring adapters, and dashboard browser verification are separate
runtime validation tasks.
