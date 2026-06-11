# DVRL World-Class Improvements

15 high-impact improvements to elevate APG Data Virtualization to production-grade, enterprise-ready quality.

---

## 1. Async-First Service Layer

**Problem**: All current methods are synchronous. Real federation involves I/O-bound connector calls (SQL, REST, object-store) that block threads.

**Solution**: Add a full `async` mirror of every core method using `asyncio`. Connectors await their I/O, multiple source fetches run concurrently via `asyncio.gather`, and the service layer never blocks the event loop.

**Impact**: 10–50x throughput on multi-source federated queries; enables FastAPI and async WSGI integration without thread pools.

---

## 2. Distributed Query Plan Optimizer

**Problem**: `push_down_optimise` does naive keyword detection with a fixed 15%/step saving estimate. It doesn't model actual connector capabilities or cardinalities.

**Solution**: Build a cost-based optimizer that inspects per-connector capability flags (`supports_predicate_pushdown`, `supports_aggregation`, `max_rows_before_scan`), estimates selectivity from column stats, and emits an ordered execution plan with per-step cost bounds.

**Impact**: Avoids full-table scans on large connectors; reduces cross-wire data movement by 2–5x in typical analytics workloads.

---

## 3. Columnar Result Streaming with Backpressure

**Problem**: `execute_query` returns results as `list[dict]` — everything materialized in memory. Large result sets (>100k rows) OOM the process.

**Solution**: Return an async generator yielding Apache Arrow `RecordBatch` chunks. Consumers control flow via async iteration; the service applies backpressure at the connector read boundary.

**Impact**: Enables unlimited result sizes, direct Parquet/CSV streaming to object store, and zero-copy handoff to downstream processors (DuckDB, Polars, Bytewax).

---

## 4. Schema Change Detection and Drift Alerts

**Problem**: `refresh_schema` stores a snapshot but never compares it against a previous snapshot to detect breaking changes (dropped columns, type widening, new NOT NULL constraints).

**Solution**: On every schema refresh, diff the new schema against the stored previous version. Emit `schema.drift_detected` audit events with a typed diff payload (`column_added`, `column_dropped`, `type_changed`). Surface breaking diffs in the pending-review queue.

**Impact**: Prevents silent query failures caused by upstream schema evolution; closes a critical data reliability gap.

---

## 5. Column-Level Lineage Graph

**Problem**: `data_lineage` captures table and transformation step names as flat string lists. No causal graph is built, so impact analysis requires scanning all lineage records manually.

**Solution**: Store lineage as a directed acyclic graph (DAG) keyed on `(source_column, virtual_column)` edges with transformation labels. Expose `get_lineage_subgraph(tenant_id, column_fqn)` that returns upstream and downstream columns transitively.

**Impact**: Enables one-click impact analysis ("which dashboards break if I drop this column?"), a mandatory control for GDPR Article 30 and SOC 2 CC6.

---

## 6. Tenant-Isolated Credential Vault Integration

**Problem**: `credentials_vaulted: bool` is a flag set by the caller with no enforcement. Nothing prevents a source being registered with `credentials_vaulted=False` and live secrets in the `metadata` dict.

**Solution**: Introduce a `CredentialVaultAdapter` abstract interface. Registration validates that the vault adapter can resolve the credential reference before the record is persisted. Reject registration if the vault lookup fails.

**Impact**: Eliminates the most common credential leak vector in data platform integrations; provides a clear extension point for HashiCorp Vault, AWS Secrets Manager, and GCP Secret Manager.

---

## 7. Query Result Diff for Incremental Caching

**Problem**: `cache_result` stores opaque TTL metadata but provides no mechanism to serve stale-while-revalidate semantics or detect when a result has materially changed.

**Solution**: On cache miss, execute the query and compute a deterministic hash of the result set. On subsequent executions, return the cached result and schedule a background refresh. If the new result hash differs by more than a configurable threshold, emit a `cache.result_drifted` event.

**Impact**: Near-zero latency for repeat analytical queries; background freshness maintenance without user-visible staleness.

---

## 8. Policy-as-Code with OPA Integration

**Problem**: `evaluate_capability_rules` is a closed Python function. Teams cannot extend rules without modifying the capability source.

**Solution**: Add an `ExternalPolicyEngine` adapter that proxies rule evaluation to Open Policy Agent (OPA) via its REST API. Fall back to the built-in rules if OPA is unavailable. Policy bundles are versioned and can be hot-reloaded without service restart.

**Impact**: Enables security teams to own and audit policy logic independently of the data platform team; supports GitOps-driven policy deployment.

---

## 9. Federated Query Audit with Cryptographic Non-Repudiation

**Problem**: Audit events are in-memory dicts. They can be deleted, tampered with, or lost on process restart.

**Solution**: Each audit event gets a SHA-256 chained digest: `hash(event_payload + prev_event_hash)`. The chain root is published to an append-only sink (PostgreSQL, object store, or Kafka). Verification endpoint re-computes the chain and reports any gaps.

**Impact**: Meets financial services and healthcare audit log tamper-evidence requirements (PCI-DSS 10.5, HIPAA §164.312).

---

## 10. Multi-Tenant Namespace Isolation with Rate Limiting

**Problem**: All tenants share a single `DVRLLifecycleService` instance keyed by string prefix. A tenant with a runaway query loop can exhaust the shared audit event list and slow all tenants.

**Solution**: Introduce `DVRLTenantRegistry` that instantiates per-tenant `DVRLLifecycleService` with dedicated resource budgets: max concurrent queries, max audit events retained, max cache entries. Operations exceeding the budget raise `TenantQuotaExceededError`.

**Impact**: Hard isolation between tenants; prevents noisy-neighbour performance degradation; enables per-tenant SLA enforcement.

---

## 11. Semantic Layer Query Translation

**Problem**: `semantic_layer` stores metric and dimension definitions as opaque dicts but provides no query translation. Users must manually map business concepts to SQL.

**Solution**: Add `semantic_query(tenant_id, layer_id, metrics, dimensions, filters)` that translates business-level metric/dimension references into SQL using the layer definitions, then routes the result through `execute_query` with full guardrail coverage.

**Impact**: Business analysts query in business terms; SQL complexity is hidden; the same guardrails cover both raw SQL and semantic queries.

---

## 12. Connection Pool Health and Circuit Breaker

**Problem**: No connection health state is maintained between calls. A dead connector causes every query that touches it to time out individually before failing.

**Solution**: Per-connector circuit breaker: after `N` consecutive failures within a rolling window, the breaker opens and queries are immediately rejected with a `ConnectorUnavailableError` and a `retry_after` timestamp. A half-open probe runs on a background task.

**Impact**: Prevents cascading timeouts; degrades gracefully; surfaces connector health in the dashboard with time-to-recovery estimates.

---

## 13. Data Contract Enforcement on Publish

**Problem**: Virtual tables can be published with any column schema. Downstream consumers have no guaranteed contract for column presence, nullability, or type compatibility.

**Solution**: Introduce `DataContract` with required columns, type assertions, and nullability constraints. `publish_virtual_table` validates the proposed schema against any attached contract and raises `ContractViolationError` with a field-level diff if the schema does not satisfy the contract.

**Impact**: Eliminates the "it broke at 3am because the upstream changed a column type" class of incidents; enables consumer-driven contract testing.

---

## 14. Async Batch Lineage Import from dbt Artifacts

**Problem**: Organizations already have column-level lineage in dbt `manifest.json` and `catalog.json`. DVRL's lineage must be populated manually.

**Solution**: Add `import_dbt_lineage(tenant_id, manifest_path, catalog_path, actor)` that parses dbt artifacts and bulk-inserts lineage DAG edges into DVRL's lineage store, mapping dbt models to registered virtual tables by name.

**Impact**: Zero-friction lineage bootstrap for dbt-using organizations; eliminates duplicate lineage maintenance.

---

## 15. Real-Time Virtualisation Metrics with OpenTelemetry Export

**Problem**: `virtualisation_analytics` computes metrics on demand by scanning in-memory collections. There is no continuous metrics stream and no integration with observability platforms.

**Solution**: Instrument every service method with OpenTelemetry span and counter exports. Expose a `metrics_snapshot` endpoint returning current gauge values in Prometheus exposition format. Spans include tenant ID, operation name, decision, and matched rules as attributes.

**Impact**: DVRL becomes fully observable; teams can set SLO alerts on query denial rate, cache hit rate, and connector latency without manual polling.
