# Cache Management (CACH) — User Guide

**Capability ID**: `cach` | **Domain**: `common` | **Version**: `1.1.0`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## Overview

CACH is APG's multi-tier cache governance and runtime capability.  It gives
generated applications a tenant-aware way to register cache namespaces, enforce
entry admission rules, manage warming and eviction, publish UI metadata, and
connect to cache backends without hard-coding a specific store.

The v1.1 release adds 19 new async methods covering stale-while-revalidate,
adaptive TTL, schema versioning, cascading tag invalidation, streaming warm,
monetary value caching, XFetch stampede protection, tenant quota enforcement,
tier statistics, and write-behind mode.

---

## Installation

```bash
pip install apg-common-cach
```

---

## Quick Start

```python
from capabilities.common.cach.service import CacheService

svc = CacheService(actor_id="api-server", tenant_id="acme")

# Register a namespace
await svc.namespace_create("products", max_entries=50_000, default_ttl_seconds=600)

# Write and read
await svc.cache_set("products", "prod:1", {"name": "Widget", "price": 9.99}, ttl_seconds=600)
result = await svc.cache_get("products", "prod:1")
assert result["hit"] is True
assert result["value"]["name"] == "Widget"
```

---

## Core Operations

### cache_set

```python
result = await svc.cache_set(
    namespace="orders",
    key="order:1001",
    value={"id": 1001, "status": "pending"},
    ttl_seconds=900,
    tags=["user:42", "order"],
)
# result: {key, namespace, stored=True, expires_at}
```

### cache_get

```python
r = await svc.cache_get("orders", "order:1001")
if r["hit"]:
    print(r["value"], r["expires_at"])
else:
    print("cache miss — fetch from source")
```

### cache_delete / cache_exists

```python
await svc.cache_delete("orders", "order:1001")
exists = await svc.cache_exists("orders", "order:1002")
print(exists["exists"])
```

### bulk_set / bulk_get

```python
await svc.bulk_set("products", {"p:1": {...}, "p:2": {...}}, ttl_seconds=600)
result = await svc.bulk_get("products", ["p:1", "p:2", "p:3"])
# result: {results: {key: value|None}, hit_count, miss_count}
```

### cache_flush

```python
await svc.cache_flush("products")       # flush one namespace
await svc.cache_flush()                 # flush all tenant namespaces
```

---

## TTL Management

### ttl_update

```python
await svc.ttl_update("session", "sess:abc", ttl_seconds=1800)
```

### expire_soon_report

```python
report = await svc.expire_soon_report("session", within_seconds=120)
# report["expiring_soon"] — list of {key, expires_at} sorted by expiry
```

---

## Namespace Governance

```python
await svc.namespace_create("reports", max_entries=10_000, default_ttl_seconds=3600)
await svc.eviction_policy("reports", "lru")
namespaces = await svc.namespace_list()
await svc.namespace_delete("reports")
```

---

## Distributed Locking

```python
lock = await svc.distributed_lock("billing-job", owner="worker-1", ttl_seconds=30)
if lock["acquired"]:
    try:
        ...  # critical section
    finally:
        await svc.distributed_unlock("billing-job", owner="worker-1")
```

---

## Tag Invalidation

### Flat tag invalidation

```python
await svc.tag_invalidate("catalog", "user:42")
```

### Cascading tag invalidation (v1.1)

```python
# Register hierarchy once during startup
await svc.register_tag_hierarchy("catalog", "user:42", [
    "user:42:orders",
    "user:42:profile",
    "user:42:preferences",
])

# Invalidate parent — cascades to all descendants via BFS
result = await svc.tag_invalidate_cascade("catalog", "user:42")
print(result["total_invalidated"], "entries removed")
print(result["tags_resolved"])   # full list of tags that were walked
```

---

## Stale-While-Revalidate (SWR) — v1.1

Serve stale data immediately while triggering async background refresh.
Eliminates cold-read latency spikes on popular keys.

```python
# Register a refresh callback (called in background on stale hit)
async def refresh_product(namespace: str, key: str):
    return await db.get_product(key.removeprefix("prod:"))

await svc.register_refresh_callback("products", refresh_product)

# Write with SWR grace window
await svc.cache_set_swr(
    "products", "prod:42", product_data,
    ttl_seconds=300, swr_grace_seconds=60,
)

# Read — if expired but within grace window: returns stale=True and refreshes
r = await svc.cache_get_swr("products", "prod:42")
if r["hit"] and r["stale"]:
    print("serving stale value while background refresh runs")
```

---

## Adaptive TTL — v1.1

Extend TTL for hot keys; let cold keys expire naturally.

```python
await svc.adaptive_ttl_configure(
    "session",
    ttl_min_seconds=120,
    ttl_max_seconds=7200,
    growth_factor=1.5,
)

# Each hit extends remaining TTL by 1.5x (capped at 7200s)
r = await svc.cache_get_adaptive("session", "user:99")
print(r.get("ttl_extended_to_seconds"))   # None if not extended
```

---

## Schema-Versioned Cache — v1.1

Prevent stale-schema cache poisoning across deployments.

```python
# Write with explicit schema version
await svc.cache_set_versioned(
    "orders", "order:1001", order_payload_v2,
    schema_version="2", ttl_seconds=600,
)

# Read — schema mismatch auto-invalidates and returns hit=False
r = await svc.cache_get_versioned("orders", "order:1001", expected_version="2")
assert r["version_mismatch"] is False

# Old reader expecting v1 gets a cache miss (safe invalidation)
r_old = await svc.cache_get_versioned("orders", "order:1001", expected_version="1")
assert r_old["version_mismatch"] is True

# Report version distribution across namespace
report = await svc.schema_version_report("orders")
print(report["version_distribution"])  # {"2": 150, "1": 3}
print(report["unversioned_entries"])
```

---

## Streaming Cache Warm — v1.1

Warm from large datasets without memory spikes.

```python
async def product_cursor():
    async for row in db.execute("SELECT id, data FROM products"):
        yield (f"prod:{row['id']}", row['data'])

async def on_progress(loaded: int, failed: int, elapsed_ms: float):
    print(f"{loaded} loaded, {failed} failed — {elapsed_ms:.0f}ms")

result = await svc.warm_cache_stream(
    "products",
    product_cursor(),
    ttl_seconds=3600,
    batch_size=200,
    progress_callback=on_progress,
)
print(result["loaded"], "entries warmed in", result["elapsed_ms"], "ms")
```

---

## Monetary Value Cache — v1.1

Store `Decimal` monetary amounts without floating-point precision loss.

```python
from decimal import Decimal

await svc.cache_set_money("pricing", "price:USD:prod42", Decimal("19.99"), "USD")

r = await svc.cache_get_money("pricing", "price:USD:prod42")
assert isinstance(r["amount"], Decimal)
assert r["currency"] == "USD"
total = r["amount"] * Decimal("3")   # Decimal arithmetic preserved
```

---

## XFetch Stampede Protection — v1.1

Probabilistic early-expiry (Vattani et al., 2015) prevents thundering-herd
failures when a popular key expires under high concurrency.

```python
# beta=1.0 standard; >1 more aggressive early recompute; <1 less aggressive
r = await svc.xfetch_get("products", "prod:42", beta=1.0)
if not r["hit"]:
    # Safe to recompute — requests are spread over time before hard expiry
    fresh_data = await source.fetch("prod:42")
    await svc.cache_set("products", "prod:42", fresh_data, ttl_seconds=600)
```

`r["early_miss"]` is `True` when XFetch triggered a probabilistic miss before
actual expiry.

---

## Tenant Quota Enforcement — v1.1

Multi-tenant SaaS requires per-tenant memory limits.

```python
# Configure per-tenant quotas
await svc.set_tenant_quota(
    "acme",
    soft_bytes=50_000_000,    # 50 MB — emits audit warning
    hard_bytes=100_000_000,   # 100 MB — blocks writes with PermissionError
)

# Report current utilisation
report = await svc.quota_usage_report("acme")
print(report["estimated_bytes"], "/", report["hard_bytes"])
print(report["utilisation_pct"], "%")
```

---

## Write-Behind Mode — v1.1

Decouple cache writes from backend persistence to reduce write latency.

```python
# Writes land in cache immediately; backend write enqueued
await svc.cache_set_write_behind("orders", "order:9999", order_data, ttl_seconds=900)
print("written to cache in microseconds")

# Drain queue in background task or scheduler
async def persist_to_db(namespace: str, key: str, value: Any):
    await db.upsert(namespace, key, value)

result = await svc.write_behind_flush(backend_fn=persist_to_db)
print(result["flushed"], "persisted,", result["remaining"], "queued")
```

---

## Tier Statistics — v1.1

Inspect L1 (hot) vs L2 (warm/cold) entry distribution.

```python
stats = await svc.tier_stats("products")
print(stats["l1_hot_entries"])      # access_count >= 5
print(stats["l2_warm_entries"])     # access_count < 5
print(stats["overall_hit_rate"])
```

---

## Analytics and Reporting

```python
# Performance dashboard
report = await svc.performance_report()

# Per-namespace statistics
stats = await svc.cache_stats("orders")

# Miss report — top-N missed keys
misses = await svc.cache_miss_report("products")

# Access frequency — top-N hottest keys
freq = await svc.access_frequency_report("products", top_n=10)

# Keys expiring soon
soon = await svc.expire_soon_report("session", within_seconds=300)

# Tier breakdown
tiers = await svc.tier_stats("products")

# Schema version distribution
versions = await svc.schema_version_report("orders")

# Full dashboard KPIs
dash = await svc.dashboard()
```

---

## Governance and Compliance

```python
# Governance summary (eviction policies, namespace count)
gov = await svc.governance_report()

# Compliance report (GDPR, PCI-DSS, HIPAA)
comp = await svc.compliance_report(framework="GDPR")

# Audit trail
events = await svc.audit_trail()
cache_hits_only = await svc.audit_trail(event_type="cache_hit")

# Tenant quota utilisation
quota_report = await svc.quota_usage_report()
```

---

## Health Monitoring

```python
health = await svc.cache_health()
print(health["status"])           # "healthy"
print(health["active_entries"])
print(health["active_locks"])
print(health["error_count"])
```

---

## Pub/Sub Notifications

```python
# Publish an invalidation event
await svc.pub_sub_notify("cache-events", {"type": "invalidation", "tag": "user:42"})

# Subscribe to receive messages since a given timestamp
messages = await svc.pub_sub_subscribe("cache-events", since="2026-06-01T00:00:00")
```

---

## Eviction

```python
# Auto-evict LFU entries when namespace exceeds max_entries
result = await svc.auto_evict("products", max_entries=5000)
print(result["evicted"], "entries removed")

# Manual consistency check — purge expired entries
check = await svc.consistency_check("products")
print(check["purged_stale_entries"])
```

---

## Import and Adapter Boundary

```python
# Control-plane helpers (governance records)
from capabilities.common.cach.api import (
    create_namespace_record,
    write_cache_entry_record,
    read_cache_entry_record,
    request_eviction_review,
    decide_eviction_review,
    register_cache_agent,
    validate_cache_lifecycle_batch,
)

# Runtime service (in-process cache + all new methods)
from capabilities.common.cach.service import CacheService
```

Production adapters should:

1. Call the control plane for a namespace policy decision.
2. Store data in the selected backend only when the decision allows it.
3. Emit audit events through APG adapters.
4. Preserve tenant isolation in backend key construction.
5. Honour encryption, TTL, invalidation, freshness, and eviction decisions.

---

## Permissions

| Permission | Grants |
|---|---|
| `cach:view` | Read dashboard and overview |
| `cach:read` | Read cache entries |
| `cach:write` | Write cache entries |
| `cach:delete` | Delete cache entries |
| `cach:manage_namespaces` | Create, configure, and delete namespaces |
| `cach:manage_policies` | Set eviction and TTL policies |
| `cach:warm` | Trigger cache warming and streaming warm |
| `cach:review_eviction` | Approve or deny eviction reviews |
| `cach:view_analytics` | Access performance and tier reports |
| `cach:manage_agents` | Register and manage cache agents |
| `cach:admin` | Full administrative access |

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/cach/dashboard` | `cach:view` | Overview |
| `/cach/namespaces` | `cach:manage_namespaces` | Operations |
| `/cach/entries` | `cach:read` | Operations |
| `/cach/policies` | `cach:manage_policies` | Governance |
| `/cach/warming` | `cach:warm` | Operations |
| `/cach/evictions` | `cach:review_eviction` | Governance |
| `/cach/hierarchy` | `cach:view` | Architecture |
| `/cach/analytics` | `cach:view_analytics` | Intelligence |

---

## Further Reading

- `service.py` — 63-method async runtime implementation
- `models.py` — Pydantic v2 data models
- `api.py` — FastAPI REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — rule engine, guardrails, UI config
- `README.md` — Quick reference and feature overview
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement proposals with full justification
- `SPECIFICATION.md` — Full functional specification
