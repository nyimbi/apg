# Graph Data Management (GRPH)

GRPH provides the APG graph foundation: tenant-scoped schemas, nodes, edges,
lineage graphs, relationship governance, bounded traversal, graph quality
inspection, first-class graph-agent composition, Bytewax lifecycle batch
governance, audit evidence, UI view models, and package metadata for generated
applications.

Use GRPH when an application needs to model connected business objects: customer
relationships, data lineage, process dependencies, knowledge graphs, service
topologies, or ownership networks. The capability is intentionally executable in
the generated Python target while remaining dependency-light enough for package
tests and offline composition.

## What GRPH Provides

- Graph schema registration for property, lineage, knowledge, and dependency graphs.
- Tenant-isolated node and edge lifecycle operations (create, update, delete, bulk).
- Schema-constrained node types and edge types with JSON Schema property validation.
- Relationship classification, RBAC review gates, and restricted-edge controls.
- Bounded traversal, lineage path queries, shortest path (BFS), and weighted path (Dijkstra).
- Community detection via label propagation.
- Node centrality scoring (degree, betweenness-approx, closeness-approx).
- Cycle detection (DFS) and structural graph diff between schema subgraphs.
- Subgraph extraction and schema-to-schema merge with conflict strategy.
- GraphML import/export for interoperability with external tools.
- Aggregate graph analytics: density, average degree, orphan count, type distributions.
- Temporal graph query returning nodes/edges created after a given timestamp.
- Pattern matching on node type and property values.
- Quality reporting for orphan nodes, missing owners, restricted edges, and graph health.
- Durable pending-review records with matched rules and review reasons.
- Provider-neutral AI graph-agent registration (Codex, Claude Code, opencode, Pi).
- Bytewax-only lifecycle batch validation.
- Audit events for all graph mutations, traversal decisions, and state changes.
- UI route metadata and view models for graph screens.
- Bytewax adapter evidence for streamed mutation and quality events.

## Runtime Surfaces

| File | Purpose |
|---|---|
| `capability_contract.py` | Configuration, deterministic rules, UI routes, adapters, and theme tokens |
| `service.py` | Runtime service — all methods documented here |
| `api.py` | Dependency-light API helper functions |
| `views.py` | View models for graph screens |
| `graph_runtime.py` | Deterministic traversal and quality primitives |
| `models.py` | Pydantic v2 models for all graph entities |
| `app.py` | Package semantic model and self-test |

## Quick Start

```python
from capabilities.common.grph.service import GrphService

service = GrphService()

# 1. Create a schema
schema = service.create_schema(
    schema_id="orders-lineage",
    tenant_id="tenant-a",
    name="Orders lineage",
    graph_kind="lineage",
    node_types={"Dataset": ["name", "system"]},
    edge_types={"DERIVES_FROM": {"classification": "restricted"}},
    source_asset_id="asset://warehouse/orders",
)

# 2. Add nodes
source = service.create_node(
    node_id="orders_raw",
    tenant_id="tenant-a",
    schema_id=schema["id"],
    node_type="Dataset",
    owner_id="data-owner",
    source_asset_id="asset://warehouse/orders",
)
target = service.create_node(
    node_id="orders_curated",
    tenant_id="tenant-a",
    schema_id=schema["id"],
    node_type="Dataset",
    owner_id="data-owner",
)

# 3. Connect nodes
service.create_edge(
    edge_id="orders_transform",
    tenant_id="tenant-a",
    schema_id=schema["id"],
    from_node_id=source["id"],
    to_node_id=target["id"],
    edge_type="DERIVES_FROM",
    owner_id="data-owner",
    classification="restricted",
    review_recorded=True,
)

# 4. Traverse lineage
path = service.lineage_path(
    traversal_id="orders_path",
    tenant_id="tenant-a",
    source_asset_id="asset://warehouse/orders",
    start_node_id=source["id"],
    max_depth=2,
)
```

## API Reference

### Core Sync Methods

| Method | Description |
|---|---|
| `create_schema(...)` | Register a typed property/lineage/knowledge/dependency graph schema |
| `create_node(...)` | Add a tenant-owned node with optional labels and properties |
| `create_edge(...)` | Connect two nodes with a typed, classified edge |
| `traverse(...)` | Bounded depth-first traversal from a start node |
| `lineage_path(...)` | Asset-scoped lineage traversal |
| `impact_analysis(...)` | Downstream impact traversal (max_depth=3 default) |
| `neighborhood(...)` | Single-hop neighbourhood view |
| `quality_report(...)` | Orphan node, missing owner, and restricted edge audit |
| `retire_schema(...)` | Retire a schema with review evidence |
| `register_graph_agent(...)` | Register an AI graph-steward agent |
| `validate_grph_lifecycle_batch(...)` | Bytewax lifecycle batch validation |
| `create_record(...)` | Compatibility helper for generated package probes |
| `dashboard_summary(...)` | Aggregate counts and pending-review queues |
| `list_schemas/nodes/edges/traversals/quality_reports/graph_agents(...)` | Tenant-filtered list queries |

### Async Methods (v2.0)

| Method | Description |
|---|---|
| `node_create(...)` | Async wrapper for `create_node` |
| `node_update(node_id, tenant_id, properties, labels)` | Merge-update properties and labels in place |
| `node_delete(node_id, tenant_id, cascade_edges)` | Delete node, optionally cascade connected edges |
| `edge_create(...)` | Async wrapper for `create_edge` |
| `edge_update(edge_id, tenant_id, properties)` | Merge-update edge properties |
| `edge_delete(edge_id, tenant_id)` | Remove a single edge |
| `shortest_path(traversal_id, tenant_id, source_node_id, target_node_id, max_depth)` | BFS shortest path |
| `weighted_path(traversal_id, tenant_id, source_node_id, target_node_id, weight_property)` | Dijkstra weighted path |
| `community_detect(report_id, tenant_id, schema_id, algorithm)` | Label-propagation community detection |
| `centrality(report_id, tenant_id, schema_id, algorithm)` | Degree / betweenness-approx / closeness-approx scores |
| `cycle_detect(report_id, tenant_id, schema_id)` | DFS cycle detection |
| `subgraph_extract(subgraph_id, tenant_id, node_ids)` | Extract induced subgraph |
| `graph_merge(merge_id, tenant_id, source_schema_id, target_schema_id, conflict_strategy)` | Merge two schema subgraphs |
| `graph_diff(diff_id, tenant_id, schema_id_a, schema_id_b)` | Structural diff of two subgraphs |
| `graph_analytics(tenant_id, schema_id)` | Density, degree distribution, type counts |
| `pattern_match(match_id, tenant_id, schema_id, node_type, edge_type, property_filter)` | Structural pattern matching |
| `temporal_graph(tenant_id, schema_id, since_iso)` | Nodes and edges created after a timestamp |
| `import_graphml(import_id, tenant_id, schema_id, graphml_content, owner_id)` | Import from GraphML XML |
| `export_graphml(tenant_id, schema_id)` | Export to GraphML XML string |
| `bulk_create_nodes(tenant_id, schema_id, nodes, owner_id)` | Batch node creation |
| `bulk_create_edges(tenant_id, schema_id, edges, owner_id)` | Batch edge creation |
| `health_check(tenant_id)` | Service health and live statistics |

## New Methods — Usage Examples

### Shortest Path (BFS)

```python
import asyncio

result = asyncio.run(service.shortest_path(
    traversal_id="path-001",
    tenant_id="tenant-a",
    source_node_id="orders_raw",
    target_node_id="orders_curated",
    max_depth=5,
))
# result["found"] -> True
# result["path_length"] -> 1
# result["node_ids"] -> ["orders_raw", "orders_curated"]
```

### Community Detection

```python
report = asyncio.run(service.community_detect(
    report_id="comm-001",
    tenant_id="tenant-a",
    schema_id="orders-lineage",
    algorithm="label_propagation",
))
# report["community_count"] -> int
# report["communities"] -> {"node_id_as_label": ["node_id", ...], ...}
```

### Graph Analytics

```python
analytics = asyncio.run(service.graph_analytics(
    tenant_id="tenant-a",
    schema_id="orders-lineage",
))
# {
#   "node_count": 2, "edge_count": 1,
#   "avg_degree": 1.0, "graph_density": 1.0,
#   "orphan_count": 0,
#   "node_type_distribution": {"Dataset": 2},
#   "edge_type_distribution": {"DERIVES_FROM": 1},
# }
```

### Bulk Ingestion

```python
nodes_created = asyncio.run(service.bulk_create_nodes(
    tenant_id="tenant-a",
    schema_id="orders-lineage",
    nodes=[
        {"id": "node-1", "node_type": "Dataset", "properties": {"name": "raw"}},
        {"id": "node-2", "node_type": "Dataset", "properties": {"name": "curated"}},
    ],
    owner_id="data-owner",
))
# returns list[dict] — one record per created node
```

### GraphML Round-Trip

```python
graphml = asyncio.run(service.export_graphml(
    tenant_id="tenant-a",
    schema_id="orders-lineage",
))

report = asyncio.run(service.import_graphml(
    import_id="import-001",
    tenant_id="tenant-b",
    schema_id="cloned-schema",
    graphml_content=graphml,
    owner_id="data-owner",
))
# report["imported_nodes"], report["imported_edges"]
```

## World-Class Enhancements (v2.0)

Fifteen targeted improvements planned for the next release cycle, in priority order:

| # | Enhancement | Impact |
|---|---|---|
| 1 | **Native Cypher Query Engine** — Lark-grammar DSL compiling `MATCH/WHERE/RETURN` to traversal primitives | Declarative ad-hoc analytics, Neo4j parity |
| 2 | **PostgreSQL + Apache AGE backend** — `StorageAdapter` protocol with `InMemoryAdapter` and `AGEAdapter`; ACID guarantees, AGE graph namespaces per tenant | Durable, restartable graphs |
| 3 | **Bytewax mutation log** — Every `_record_event` published as a CloudEvent to `grph.mutations.<tenant_id>` via bytewax stream processor | Real-time downstream consumers, projection materialisation |
| 4 | **Incremental PageRank** — Power-iteration with damping `d=0.85`, convergence `epsilon`, dirty-flag caching | Reliable hub/authority scores |
| 5 | **Semantic edge inference via Ollama** — Embed node properties with `nomic-embed-text`, create `SEMANTICALLY_SIMILAR` edges above cosine threshold | Automatic knowledge-graph enrichment |
| 6 | **Bi-temporal versioning** — `valid_from`, `valid_to`, `transaction_time` on nodes/edges; `as_of(ts)` and `between(t1, t2)` queries | Compliance, lineage auditing, time-series |
| 7 | **GNN feature extraction via PyTorch Geometric** — 2-layer GCN producing 64-dim node embeddings stored as `properties["__gnn_embedding"]` | Searchable node representations via SRCH |
| 8 | **Hierarchical partitioning (METIS-style)** — Recursive spectral bisection via Fiedler eigenvector; partition tree for cross-shard traversal | Sub-linear analytics on large graphs |
| 9 | **Graph compression** — `compress_schema` replaces structurally-equivalent nodes with superposition nodes; 40–80% storage reduction | Dense graph scalability |
| 10 | **Schema validation + migration engine** — JSON Schema per node/edge type enforced at write time; `migrate_schema` applies transformation specs atomically | Data quality enforcement |
| 11 | **Distributed read replicas** — Redis-backed `GraphReadCache` with Bytewax invalidation events; horizontal read scaling with read-your-writes consistency | Multi-process deployments |
| 12 | **Full-text + vector search via SRCH** — `search_nodes(query)` runs BM25 + vector hybrid retrieval; GRPH as knowledge-graph RAG backend | Fuzzy and semantic node discovery |
| 13 | **CRDT merge semantics** — G-Set/2P-Set node and edge state; `merge(snapshot_a, snapshot_b)` without conflicts | Offline/edge/air-gapped deployments |
| 14 | **Interactive visualisation export** — `export_vis_json` (Cytoscape.js) and `export_d3_force` (D3 force-directed); served at `/grph/explorer/json` | Embedded graph explorer in the FAB UI |
| 15 | **Zero-trust capability-based access control** — `CapabilityToken` (signed JWT) scoping `allowed_node_types`, `allowed_edge_types`, `max_traversal_depth`; revocation via Redis | Fine-grained, time-limited agent delegation |

## Guardrails

GRPH denies operations without tenant context, required schema/node/edge identity,
owners, source/target nodes, schema-defined types, or lineage source assets.
Review-required operations are durable: unknown schema kinds, non-allowlisted node
labels, unknown or restricted relationship classifications, self edges, deep
traversals, quality-threshold breaches, and privileged graph-agent roles are stored
with `pending_review` status plus matched rule names and review reasons. True deny
decisions still raise before mutation. Schema retirement and unaudited state changes
remain deny-only guardrails. Graph-agent registrations that use an unsupported
runtime or role, omit scope/owner/purpose, or hide machine contribution are denied.
Lifecycle batches not routed through Bytewax are denied and retained as denied batch
evidence.

## Composition

GRPH depends on **MDM, META, ETLP, SRCH, AICR, and CONF** for data-governance,
search, AI-agent, and configuration context. Optional adapters connect to **AUTH,
AUDL, MONI, CACH, KNGR**, and Bytewax-backed event streams. The generated package
composes into larger applications through its semantic model, UI manifest, API
helpers, first-class agent manifest, streaming manifest, and service runtime.

---

© 2025 Datacraft · Author: Nyimbi Odero · www.datacraft.co.ke
