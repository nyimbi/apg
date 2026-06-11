# GRPH — World-Class Improvement Roadmap

**Capability**: Graph Database (grph)
**Domain**: common
**Author**: Nyimbi Odero | Datacraft
**Date**: 2026-06-11

---

## 1. Native Cypher Query Engine

**Current gap**: All traversal and pattern matching is imperative Python. There is
no declarative query language.

**Improvement**: Embed a lightweight Cypher DSL parser (Lark grammar) that
compiles `MATCH (n:Person)-[:KNOWS]->(m) WHERE n.age > 30 RETURN m` into the
existing traversal primitives. This unlocks composable analytics, ad-hoc
reporting, and parity with Neo4j idioms that data engineers already know.

---

## 2. Persistent Storage Backend via PostgreSQL + Apache AGE

**Current gap**: All state lives in in-process Python dicts. A process restart
wipes the entire graph.

**Improvement**: Introduce a `StorageAdapter` protocol with two implementations:
`InMemoryAdapter` (current) and `AGEAdapter` that writes nodes/edges to a
PostgreSQL 16 instance with the Apache AGE extension installed. The AGE extension
exposes a native property-graph layer over SQL tables, giving ACID guarantees,
full-text search, and Cypher execution through `ag_catalog.cypher()`. Tenant
isolation maps to AGE graph namespaces.

---

## 3. Streaming Mutation Log via Apache Kafka / Redpanda

**Current gap**: Audit events are stored in-process with no fanout. Downstream
consumers cannot react to graph changes in real time.

**Improvement**: Publish every `_record_event` call to a Kafka topic
`grph.mutations.<tenant_id>` using `aiokafka`. Include the full entity diff as a
CloudEvent envelope. Bytewax consumers can then materialise read-optimised
projections (e.g. adjacency matrices, inverted label indices) without touching
the primary store.

---

## 4. Incremental PageRank with Convergence Guarantee

**Current gap**: The `centrality` method computes only degree and crude
approximations. There is no eigenvector-based ranking.

**Improvement**: Implement iterative PageRank (power iteration with damping
factor `d=0.85`) that runs until `max(|r_new - r_old|) < epsilon` or a
configurable iteration cap. Expose `damping_factor` and `max_iterations`
parameters. Cache results with a dirty flag so re-runs only process changed
nodes. This gives reliable hub/authority scores for knowledge graphs and
influence networks.

---

## 5. Semantic Similarity Edge Inference via Ollama Embeddings

**Current gap**: Edges must be explicitly created. There is no latent-structure
discovery.

**Improvement**: Add an async `infer_semantic_edges` method that embeds node
property values (via the local Ollama `nomic-embed-text` model), computes cosine
similarity between all node-pairs, and creates `SEMANTICALLY_SIMILAR` edges for
pairs above a configurable threshold. This enables automatic knowledge-graph
enrichment without manual curation.

---

## 6. Temporal Versioning with Bi-temporal Node/Edge Records

**Current gap**: The `temporal_graph` method is a hack that uses audit-event IDs
as a proxy for time. No true valid-time or transaction-time is stored.

**Improvement**: Add `valid_from: datetime`, `valid_to: datetime | None`, and
`transaction_time: datetime` fields to `GraphNode` and `GraphEdge`. Expose
`as_of(timestamp)` and `between(t1, t2)` queries that reconstruct the graph
state at any point in time. This is essential for compliance, lineage auditing,
and time-series analytics.

---

## 7. Graph Neural Network Feature Extraction via PyTorch Geometric

**Current gap**: GRPH produces structural metrics but no learnable representations.

**Improvement**: Add an optional `gnn_features` method that converts the
in-memory graph to a `torch_geometric.data.Data` object and runs a 2-layer GCN
forward pass to produce 64-dimensional node embeddings. Embeddings are stored
as `properties["__gnn_embedding"]` and made searchable via the `srch` capability.
The GCN weights are loaded from a pre-trained Ollama-served GGUF model.

---

## 8. Hierarchical Graph Partitioning (METIS-style)

**Current gap**: Large graphs are traversed in full. There is no recursive
decomposition for divide-and-conquer analytics.

**Improvement**: Implement recursive spectral bisection that splits the graph
into balanced partitions by finding the Fiedler eigenvector of the graph
Laplacian (approximated via power iteration). The resulting partition tree is
stored as a nested `subgraph_id -> children` mapping and used to accelerate
community detection and cross-shard traversal planning.

---

## 9. Graph Compression via Edge Bundling and Node Superposition

**Current gap**: Dense graphs with millions of nodes are stored and traversed
naively. Memory usage is O(N + E) with large constants.

**Improvement**: Add a `compress_schema` method that identifies structurally
equivalent nodes (same type, same neighbour multiset) and replaces them with
superposition nodes carrying a `multiplicity` weight. Reduces storage by 40-80%
on typical enterprise graphs (e.g. audit trails, IoT topologies). Decompression
is lazy and transparent to callers.

---

## 10. Property-Graph Schema Validation and Migration Engine

**Current gap**: `GraphSchema.node_types` and `edge_types` are plain dicts with
no enforcement at write time beyond binary present/absent checks.

**Improvement**: Accept a JSON Schema fragment per node/edge type. On every
`create_node` / `create_edge` call, validate `properties` against the schema and
collect violations into `review_reasons`. Add a `migrate_schema` method that
applies a transformation spec (rename fields, add defaults, drop deprecated keys)
across all existing nodes and edges in a single atomic pass, emitting a migration
audit event with before/after counts.

---

## 11. Distributed Read Replicas via Read-Your-Writes Cache

**Current gap**: `GrphService` is a singleton. Multi-process deployments diverge
silently.

**Improvement**: Extract a `GraphReadCache` backed by Redis (or `diskcache` for
local-only) that is populated on write and invalidated on retire/delete. All
`list_*` and `dashboard_summary` calls read from cache first. A
`cache_invalidation_event` is emitted to the Kafka mutation log so that peer
replicas can self-invalidate. This enables horizontal read scaling with
read-your-writes consistency.

---

## 12. Full-Text and Vector Search Integration with SRCH Capability

**Current gap**: `pattern_match` only supports exact property equality. There is
no fuzzy, full-text, or semantic search over node properties.

**Improvement**: Integrate with the `srch` capability to index every node's
`properties` dict at creation time. Expose an async `search_nodes` method
accepting a query string that runs through SRCH's BM25 + vector hybrid retrieval
pipeline and returns ranked node matches with relevance scores. Enable GRPH to
serve as the backend for knowledge-graph–powered RAG pipelines.

---

## 13. Conflict-Free Replicated Graph (CRDT) for Offline/Edge Deployment

**Current gap**: Edge mutations are last-write-wins with no merge semantics.
Offline or disconnected agents overwrite each other's changes.

**Improvement**: Model node and edge state as add-only grow-sets (G-Sets) for
creation events and two-phase sets (2P-Sets) for removals. Implement a `merge`
CRDT operation that takes two `GrphService` snapshots and produces a consistent
union without conflicts. This is critical for field devices, air-gapped
deployments, and multi-region active-active setups.

---

## 14. Interactive Graph Visualisation Export (D3 / Cytoscape.js)

**Current gap**: GRPH produces dict payloads with no rendering layer.

**Improvement**: Add an `export_vis_json` method that emits a Cytoscape.js-
compatible JSON bundle (`elements.nodes`, `elements.edges` with full property
payloads and computed centrality scores as visual size/colour hints). Add a
`export_d3_force` method for D3 force-directed layouts. Both are served through
the Flask-AppBuilder blueprint at `/grph/explorer/json` and consumed by the
embedded Cytoscape.js widget in the UI.

---

## 15. Zero-Trust Capability-Based Access Control at the Edge Level

**Current gap**: Tenant isolation is enforced but there is no sub-tenant,
attribute-based, or capability-based policy layer. Any actor with a valid
`tenant_id` can read and write any node/edge in that tenant.

**Improvement**: Introduce a `CapabilityToken` model (signed JWT with claims
`{tenant_id, actor_id, allowed_node_types: [], allowed_edge_types: [],
max_traversal_depth: int, expiry: datetime}`) verified on every service call.
Tokens are issued by the `auth` capability and revocable via a Redis revocation
list. This enables fine-grained, time-limited delegation: a graph-agent can be
restricted to reading only `Dataset` nodes up to depth 3 without any code
changes.
