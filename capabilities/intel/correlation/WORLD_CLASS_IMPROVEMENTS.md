# World-Class Improvements: intel_correlation

**Capability**: Event Correlation (`intel_correlation`)
**Domain**: Intel
**Scope**: Multi-source event correlation, timeline construction, attribution

---

## 1. Streaming Correlation Pipeline (Bytewax Integration)

Current batch processing blocks until all pairs are evaluated. Replace with a Bytewax dataflow that emits correlation results as a stream so downstream consumers receive partial results immediately. Use `bytewax.dataflow.Dataflow` with a windowed `collect` operator keyed on entity_type. This gives sub-second latency on high-volume feeds.

## 2. Graph-Native Entity Resolution with NetworkX / igraph

The current matrix approach (`correlation_matrix`) is O(n²) in memory and recomputes from scratch. Replace with an incremental adjacency list backed by igraph or NetworkX. Store the graph persistently alongside `self.entities`. Graph traversal (BFS/DFS, PageRank, community detection) then runs in O(E) instead of O(n²).

## 3. Jaccard + MinHash Approximate Similarity for Text References

`source_entity_overlap` uses exact set intersection. For high-cardinality string references (IP addresses, hashes, names with typos) add a MinHash LSH index (e.g. `datasketch`) so near-duplicate references are matched with configurable false-positive rate. Dramatically improves recall for dirty data.

## 4. Temporal Pattern Mining with TFIDF-like Burst Detection

`temporal_correlation` bins events but ignores burst anomalies. Add a sliding-window burst detector: compute the mean and stddev of bin counts over the last N windows and flag bins where count > mean + 2σ as correlated bursts. Returns burst windows annotated with z-scores, enabling anomaly-driven correlation.

## 5. Attribution Chain Tracking (Kill-Chain Mapping)

Add an `attribution_chain` method that, given a root observation, walks forward through time-ordered observations on linked entities to reconstruct a kill-chain or causal sequence. Each node carries a `phase` label (e.g. MITRE ATT&CK tactic) and confidence decay. Output is a DAG suitable for threat-report generation.

## 6. Adaptive Confidence Decay (Staleness Weighting)

`confidence_propagate` applies a fixed decay per hop. Extend this with time-based staleness: observations older than a configurable TTL (e.g. 90d) receive an additional multiplicative penalty derived from an exponential decay function `exp(-λ·age_days)`. This prevents stale observations from inflating confidence scores.

## 7. Cross-Tenant Federated Correlation (Privacy-Preserving)

Add a `federated_correlation` method that uses Private Set Intersection (PSI) or differential-privacy noise to compute entity overlap across tenants without exposing raw references. Results contain only overlap counts and similarity scores, never raw identifiers. Controlled by a `federation_authority_id` parameter requiring pre-recorded authority.

## 8. Causal Bayesian Network Scoring

Extend `predictive_correlation` from a simple linear trend to a Bayesian network where observation types are nodes and co-occurrence frequencies define conditional probabilities. Use `pgmpy` or a custom CPT table. This gives calibrated probability estimates (not just trend direction) and supports what-if queries.

## 9. Anomaly Detection on Correlation Score Distributions

Add a `correlation_anomaly_detect` method that fits a Gaussian Mixture Model (GMM) on the historical distribution of correlation scores and flags new scores that fall in low-density regions as anomalous. Useful for detecting novel attack patterns that produce unusual correlation signatures not seen before.

## 10. Rule Hot-Reload Without Service Restart

`CorrelationRule` objects are currently immutable once stored. Add a `reload_rule` method that atomically swaps a rule's `threshold_score` and `rule_reference` and invalidates the `_matrix_cache` entries depending on that rule. Exposes a webhook endpoint so a CI pipeline can push updated rules to a running service.

## 11. Distributed Locking for Concurrent Cluster Merge

`cluster_merge` reads and writes cluster state without a lock, making it unsafe under concurrent async calls. Add a per-tenant `asyncio.Lock` map and acquire the lock keyed on `(tenant_id, cluster_id_a, cluster_id_b)` before mutation. Also emit a CloudEvent `intel.correlation.cluster.merged` to the Bytewax stream.

## 12. Explainability Layer: Correlation Rationale Generation

Add a `explain_correlation` method that, given a `correlation_id`, reconstructs a human-readable rationale string by enumerating the contributing factors (shared source, same entity type, temporal co-occurrence, spatial proximity, behavioural similarity) with per-factor weights. Output is suitable for analyst reports and audit trails.

## 13. Pagination and Cursor-Based Result Streaming

`correlation_graph_export` and `false_positive_filter` truncate results at fixed limits (50, 20 items). Replace truncation with a cursor-based API: methods accept `cursor: str | None` and `page_size: int = 100` and return `{"items": [...], "next_cursor": "..."}`. Enables clients to stream arbitrarily large result sets without memory pressure.

## 14. Persistent Storage Adapter (PostgreSQL via asyncpg)

All state lives in Python dicts, meaning restarts lose data. Abstract the dict stores behind a `CorrelationStore` protocol with a `PostgresCorrelationStore` implementation using `asyncpg` connection pools. The service constructor accepts `store: CorrelationStore | None`; if provided, all CRUD methods persist and retrieve via SQL. Schema migrations managed by Alembic.

## 15. OpenTelemetry Distributed Tracing

Instrument every async method with `opentelemetry-api` spans. Each method creates a child span named `intel_correlation.<method_name>` with attributes `tenant_id`, `correlation_id`, and `score`. This enables end-to-end latency profiling across microservices, identification of slow correlation types, and integration with Jaeger/Tempo dashboards.
