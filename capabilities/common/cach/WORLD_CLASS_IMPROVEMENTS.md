# CACH - World Class Improvements

**Capability**: Cache Management (cach) | **Domain**: common
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## 1. Stale-While-Revalidate (SWR) Pattern

**Category**: Cache Strategy

**Justification**: SWR is the most effective technique for eliminating cold-start latency in production systems. When a key is "stale" (past TTL but within a grace window), serve the stale value immediately while triggering background refresh. This decouples read latency from backend refresh latency. Industry research (Cloudflare, Fastly) shows P99 latency drops 40-60% when SWR is applied to frequently-read keys.

**Implementation**: Add `swr_grace_seconds` to `cache_set`. In `cache_get`, when a key is expired but within the grace window: return the stale value with `stale=True`, and fire a background coroutine via `create_tracked_task` to re-fetch using a registered refresh callback. The refresh callback is stored per-namespace in `_refresh_callbacks: dict[str, Callable]`.

**Competitor Reference**: Vercel/SWR library, Cloudflare Cache-Control `stale-while-revalidate` directive, Nginx `proxy_cache_revalidate`.

---

## 2. L1/L2 Tiered Cache with Promotion and Demotion

**Category**: Architecture

**Justification**: A single flat store treats a 10-byte session token the same as a 50 KB rendered HTML fragment. Tiered caches (L1 = bounded in-process, L2 = Redis-compatible) reduce P50 latency to microseconds for hot data while keeping large objects off the heap. Cache promotion on repeated L2 hits is proven by Netflix EVCache and Facebook Memcached pooling architecture.

**Implementation**: Add `CacheTier` enum (`L1_MEMORY`, `L2_REDIS`, `L3_CDN`). Store `_l1: BoundedCache` (already imported from reliability) as the hot layer; `_store` becomes L2. `cache_get` checks L1 first, promotes to L1 on L2 hit if `access_count >= promotion_threshold`. `cache_set` writes both tiers. Add `tier_stats()` method reporting per-tier hit rates.

**Competitor Reference**: Redis + local dict pattern (Netflix EVCache), Caffeine + Redis (Spring Cache), Hazelcast Near Cache.

---

## 3. Write-Through and Write-Behind Modes

**Category**: Consistency

**Justification**: Write-through (cache + backend synchronously) and write-behind (cache immediately, backend async) are the two primary consistency models for production cache deployments. Without this, CACH can only be used as a read-aside cache, which shifts cache-fill complexity to every consumer. Write-behind with batching reduces backend write IOPS by 10-100x for high-write workloads.

**Implementation**: Add `CacheWriteMode` enum (`READ_ASIDE`, `WRITE_THROUGH`, `WRITE_BEHIND`). Register a `write_backend_fn: Callable[[str, str, Any], Awaitable[None]]` per namespace. In `cache_set`, when mode is `WRITE_THROUGH`, await the backend write before confirming. For `WRITE_BEHIND`, append to a `_write_queue: asyncio.Queue` and drain in a background task with configurable batch size and flush interval.

**Competitor Reference**: Spring Cache `@CachePut`, Redis OM write-behind patterns, Hazelcast MapStore.

---

## 4. Cache Stampede (Thundering Herd) Protection via Probabilistic Early Expiry

**Category**: Reliability

**Justification**: Cache stampede is the #1 reliability failure mode in heavily-loaded caches. When a popular key expires, hundreds of simultaneous requests all miss, triggering parallel backend fetches. XFetch (Vattani et al., 2015) uses probabilistic early recomputation: recompute before expiry with probability that increases as expiry approaches. This eliminates stampedes without distributed coordination.

**Implementation**: Add `xfetch_beta: float = 1.0` to `cache_set`. In `cache_get`, compute `early_expiry = -delta * beta * ln(random())` where `delta` is the last recompute time. If `now + early_expiry > expires_at`, treat as a miss and trigger recompute. Store `last_delta_seconds` per key.

**Competitor Reference**: XFetch algorithm (Vattani et al. 2015), used in Squarespace, Etsy production cache layers.

---

## 5. Read-Your-Writes Session Consistency

**Category**: Consistency

**Justification**: In distributed systems with multiple cache nodes, a write on node A may not be visible on node B for milliseconds. Users who just wrote data ("just placed order") must read their own writes. LinkedIn, Amazon, and Facebook documented this as a top-5 user-visible correctness bug. Session consistency guarantees a reader always sees their own writes.

**Implementation**: Add `session_id` to `cache_set` and `cache_get`. Maintain `_session_writes: dict[str, dict[str, str]]` mapping `session_id -> {full_key -> version_id}`. In `cache_get`, if `session_id` provided and a write version exists, bypass cache staleness and return the version-pinned value. Version IDs use `uuid7str()` for monotonic ordering.

**Competitor Reference**: Amazon DynamoDB Session Consistency, Cassandra `LOCAL_QUORUM`, LinkedIn Espresso session routing.

---

## 6. Multi-Level Tag Hierarchy with Cascading Invalidation

**Category**: Invalidation

**Justification**: Flat tags (`tag_invalidate`) work for simple cases. Real systems need hierarchical invalidation: invalidating tag `user:42` should cascade to `user:42:orders`, `user:42:profile`, `user:42:preferences`. Without hierarchy, a single entity update requires the caller to enumerate all affected tags — which is error-prone and causes stale-data bugs. This is the approach used by Symfony Cache, Craft CMS, and Wagtail.

**Implementation**: Add `tag_hierarchy: dict[str, list[str]]` to namespace config. `tag_invalidate` recursively resolves child tags via BFS traversal of the hierarchy before deleting. Add `register_tag_hierarchy(namespace, parent, children)` method. Track tag ancestry in `_tag_graph: dict[str, set[str]]`.

**Competitor Reference**: Symfony Cache tag invalidation, Varnish VCL ban expressions, Fastly surrogate keys with cascade.

---

## 7. Distributed Cache Topology Awareness with Consistent Hashing

**Category**: Distribution

**Justification**: The current implementation is single-node. Production caches must distribute keys across nodes using consistent hashing (Karger et al., 1997) to minimize reshuffling when nodes join or leave. Virtual nodes (vnodes) reduce hotspots. This allows CACH to front Redis Cluster, Memcached, or a custom node pool without the adapter knowing backend topology.

**Implementation**: Add `ConsistentHashRing` class using `hashlib.sha256` for key placement with configurable virtual nodes per real node. `CacheService` accepts `nodes: list[str]` and routes full keys to nodes via the ring. `cache_get`/`cache_set` resolve the target node before dispatching. Add `rebalance_report()` showing key distribution across nodes and predicted migration count if a node is added/removed.

**Competitor Reference**: Amazon ElastiCache cluster mode, Redis Cluster, Memcached libketama.

---

## 8. Adaptive TTL Based on Access Frequency

**Category**: Intelligence

**Justification**: Fixed TTL wastes memory for cold keys (they sit until expiry) and causes unnecessary cache misses for hot keys (they expire before they should). Adaptive TTL extends TTL on each access (up to `max_ttl`) and shrinks it for keys that haven't been accessed (down to `min_ttl`). Netflix and Instagram reported 20-30% memory savings from adaptive TTL without degrading hit rate.

**Implementation**: Add `adaptive_ttl: bool = False` to namespace config with `ttl_min_seconds`, `ttl_max_seconds`, and `ttl_growth_factor`. In `cache_get`, after a hit: `new_ttl = min(current_remaining * growth_factor, max_ttl)`, call `ttl_update`. Add `adaptive_ttl_report(namespace)` showing TTL distribution and projected expiry histogram.

**Competitor Reference**: Instagram's Cachemaker, Netflix EVCache adaptive TTL, Redis `OBJECT FREQ`.

---

## 9. Circuit Breaker for Backend Adapters

**Category**: Reliability

**Justification**: When a cache backend (Redis, Memcached) becomes unavailable, naive implementations retry indefinitely, exhausting connection pools and cascading failures. The Circuit Breaker pattern (Martin Fowler) automatically stops requests to a failing backend and allows periodic probing for recovery. `BoundedCache` from `capabilities.common.reliability` already exists — the `CircuitBreaker` is also available there.

**Implementation**: Add `_backend_circuit_breaker: CircuitBreaker` (from `capabilities.common.reliability`) to `CacheService`. Wrap all backend adapter calls in the circuit breaker. When open, fall back to L1 or raise `CacheBackendUnavailableError` with degraded-mode flag. Add `circuit_state()` method reporting current state, failure rate, and last probe time.

**Competitor Reference**: Netflix Hystrix (Resilience4j), AWS SDK retry/backoff patterns, Redis Sentinel failover.

---

## 10. Semantic Versioning for Cached Values

**Category**: Cache Invalidation

**Justification**: Schema evolution is the silent cache corruption bug. A background job updates a serialized object format; old cached values with the old schema are served to new readers that expect the new schema. This causes `KeyError`, `ValidationError`, or silent data corruption. Versioned values require readers to declare expected version; mismatches trigger invalidation. Twitter, Dropbox, and Stripe use version headers in cache values for schema safety.

**Implementation**: Add `value_schema_version: str = "1"` to `cache_set`. Store as `_schema_version` in the record. Add `cache_get_versioned(namespace, key, expected_version)` that returns `version_mismatch=True` and deletes the stale entry if versions differ. Add `schema_version_report(namespace)` showing version distribution across live entries.

**Competitor Reference**: Twitter Twemcache version tokens, Stripe cache key versioning, Dropbox DivvyDrive schema migration.

---

## 11. Cache Warming via Query Result Streaming

**Category**: Performance

**Justification**: Batch `warm_cache` loads all data into memory before writing, creating a memory spike for large datasets. Streaming warming loads and writes entries incrementally, respecting memory pressure and allowing progress reporting. Shopify and Pinterest use streaming warm-up to pre-populate caches from database cursors without OOMing application servers.

**Implementation**: Add `warm_cache_stream(namespace, source_iter, ttl_seconds, batch_size, progress_callback)` that accepts any async iterator yielding `(key, value)` tuples. Processes in `batch_size` chunks with `asyncio.sleep(0)` between batches to yield the event loop. Reports progress via callback with `(loaded, failed, elapsed_ms)`. Add `warming_progress: dict[str, _R]` to track active warm operations.

**Competitor Reference**: Redis `RESTORE` streaming, Shopify Dalli warm-up pattern, Pinterest Memcached prefill pipeline.

---

## 12. Tenant Quota Enforcement with Hard and Soft Limits

**Category**: Governance

**Justification**: Multi-tenant caches without quotas allow one noisy tenant to evict other tenants' entries. Soft limits trigger warnings; hard limits reject writes. This is table-stakes governance for any SaaS product. Salesforce, Twilio, and Stripe enforce per-tenant resource quotas at every layer, including cache.

**Implementation**: Add `quota_bytes_soft: int` and `quota_bytes_hard: int` to namespace config. In `cache_set`, compute tenant's current usage via `_estimate_tenant_bytes()` (sum of JSON-serialized value sizes). If above soft limit: emit a `quota_warning` audit event. If above hard limit: raise `TenantQuotaExceededError` without writing. Add `quota_usage_report(tenant_id)` returning usage, soft/hard limits, and utilization %.

**Competitor Reference**: Salesforce Governor Limits, Twilio rate limiting, Redis `maxmemory-policy` per-tenant via keyspace partitioning.

---

## 13. Cache Entry Encryption at Rest with Key Rotation

**Category**: Security

**Justification**: PCI-DSS, HIPAA, and GDPR require encryption of sensitive cached data. Many teams encrypt at the transport layer but leave cache values in plaintext, creating a data-at-rest exposure when cache dumps or heap snapshots are taken. Key rotation (rotating the encryption key without downtime) is required for compliance with NIST SP 800-57. The existing `CACH` spec already mentions `encryption_required` — this implements it.

**Implementation**: Add `encryption_key_id: str | None` to `cache_set`. Maintain `_encryption_keys: dict[str, bytes]` (in production, backed by a KMS adapter). Encrypt values using `cryptography.fernet.Fernet` (AES-128-CBC + HMAC-SHA256). Add `rotate_encryption_key(old_key_id, new_key_id)` that re-encrypts all entries using the old key. Add `encryption_audit_report()` showing encrypted vs plaintext entry counts per namespace.

**Competitor Reference**: AWS ElastiCache at-rest encryption, HashiCorp Vault Transit Secrets Engine, Redis Enterprise encryption.

---

## 14. Predictive Prefetching via Access Pattern Analysis

**Category**: Intelligence

**Justification**: Reactive caches (fetch-on-miss) always incur at least one cold miss per key. Predictive prefetching uses access history to pre-populate keys before they are requested. LinkedIn's Replica DB prefetching and Google's Chrome preload heuristics demonstrate 15-25% reduction in user-perceived latency from prefetching. The existing `predictive_engine.py` stub in this capability confirms this was intended.

**Implementation**: Add `_access_sequence: list[tuple[str, float]]` (key, timestamp) per namespace. After N accesses, fit a simple Markov chain: `P(key_B | key_A)`. When `key_A` is accessed and `P(key_B | key_A) > prefetch_threshold`, fire a background prefetch of `key_B` via registered `prefetch_fn`. Add `prefetch_stats()` returning predicted vs actual hit improvement, false positive rate, and prefetch queue depth.

**Competitor Reference**: LinkedIn Replica DB prefetch, Google AMP prefetch, AWS DynamoDB DAX predictive prefetch.

---

## 15. Observability: OpenTelemetry Span and Metric Export

**Category**: Observability

**Justification**: `logger` and `_audit_log` provide internal observability, but production operations teams use OpenTelemetry-compatible systems (Datadog, Grafana Tempo, Honeycomb). Without structured spans and counters, cache behavior is invisible to SREs. Every Google, Amazon, and Netflix service emits cache hit/miss/latency metrics as P0 observability requirements. The `opentelemetry-api` package has zero runtime dependencies when no exporter is configured (no-op by default).

**Implementation**: Add optional `opentelemetry-api` integration. In `cache_get`/`cache_set`/`cache_delete`, create spans via `tracer.start_as_current_span("cach.get")` with attributes `cache.hit`, `cache.namespace`, `cache.tenant_id`, `cache.key_hash` (hashed for PII safety). Export counters `cach.hits`, `cach.misses`, `cach.evictions`, `cach.latency_ms` via `opentelemetry.metrics`. Wrap in `try/except ImportError` to remain zero-dependency when `opentelemetry-api` is absent.

**Competitor Reference**: Redis OpenTelemetry exporter, AWS ElastiCache CloudWatch integration, Datadog APM cache instrumentation.
