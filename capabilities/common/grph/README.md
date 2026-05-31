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

- Graph schema registration for property, lineage, knowledge, and dependency
  graphs.
- Tenant-isolated node and edge lifecycle operations.
- Schema-constrained node types and edge types.
- Relationship classification, RBAC review gates, and restricted-edge controls.
- Bounded traversal and lineage path queries.
- Quality reporting for orphan nodes, missing owners, restricted edges, and
  graph health.
- Provider-neutral AI graph-agent registration for Codex, Claude Code,
  opencode, Pi, and future runtimes.
- Bytewax-only lifecycle batch validation for schema, node, edge, traversal,
  lineage, impact, quality, and graph-agent changes.
- Audit events for graph mutations, traversal decisions, review-required paths,
  and state changes.
- UI route metadata and view models for graph operations.
- Bytewax adapter evidence for streamed graph mutation and quality events.

## Runtime Surfaces

- `capability_contract.py` defines configuration, deterministic rules, UI
  routes, adapters, and theme tokens.
- `service.py` is the generated-app runtime service used by tests and package
  probes.
- `api.py` exposes dependency-light API helper functions.
- `views.py` exposes generated-app view models for graph screens.
- `graph_runtime.py` contains deterministic traversal and quality primitives.
- `app.py` exposes the package semantic model and self-test.

## Lifecycle

1. Create or import a graph schema.
2. Register typed nodes with owners and optional source assets.
3. Register typed edges between tenant-local nodes.
4. Run traversals, lineage paths, neighborhood views, or impact analysis.
5. Generate quality reports and review restricted relationships.
6. Register graph agents for schema, relationship, traversal, lineage, impact,
   quality, and lifecycle governance work.
7. Validate lifecycle batches through Bytewax processor policy.
8. Inspect audit events and operational metrics.
9. Retire graph schemas or relationships with review evidence.

## Example

```python
from capabilities.common.grph.service import GrphService

service = GrphService()
schema = service.create_schema(
    schema_id="orders-lineage",
    tenant_id="tenant-a",
    name="Orders lineage",
    graph_kind="lineage",
    node_types={"Dataset": ["name", "system"]},
    edge_types={"DERIVES_FROM": {"classification": "restricted"}},
    source_asset_id="asset://warehouse/orders",
)
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
path = service.lineage_path(
    traversal_id="orders_path",
    tenant_id="tenant-a",
    source_asset_id="asset://warehouse/orders",
    start_node_id=source["id"],
    max_depth=2,
)
agent = service.register_graph_agent(
    agent_id="graph-steward",
    tenant_id="tenant-a",
    name="Graph Steward",
    runtime="codex",
    role="graph_steward",
    scope="orders lineage schema and edge quality",
    owner="data-platform",
    purpose="review relationship quality and lineage impact",
)
batch = service.validate_grph_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=1,
    operation="graph_agent_batch",
)
```

## Guardrails

GRPH denies graph operations without tenant context, required schema/node/edge
identity, owners, source/target nodes, schema-defined types, or lineage source
assets. It requires review evidence for restricted edges, deep traversals,
unknown schema kinds, non-allowlisted labels/properties, high-volume mutation
batches, privileged graph-agent roles, schema retirement, and state-changing
operations without audit events. It denies graph-agent registrations that use an
unsupported runtime or role, omit scope, owner, or purpose, or hide machine
contribution. It denies lifecycle batches that are not routed through Bytewax.

## Composition

GRPH depends on MDM, META, ETLP, SRCH, AICR, and CONF for data-governance,
search, AI-agent, and configuration context. Optional adapters connect to AUTH,
AUDL, MONI, CACH, KNGR, and Bytewax-backed event streams. The generated package
can be composed into larger applications through its semantic model, UI
manifest, API helpers, first-class agent manifest, streaming manifest, and
service runtime.
