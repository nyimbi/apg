# Knowledge Graph (KNGR)

KNGR provides APG's executable knowledge-graph capability: tenant-scoped source
registration, entity resolution, evidence-backed relationship linking, semantic
enrichment, bounded reasoning paths, curation, publication, first-class
knowledge-agent composition, Bytewax lifecycle batch governance, audit evidence,
UI view models, and package metadata for generated Python applications.

Use KNGR when an application needs a curated semantic layer over business facts:
ERP master data, procurement context, customer context, regulatory evidence,
AI-agent working memory, GraphRAG context, data lineage, or operational
dependency maps. The capability composes with APG graph, NLP, metadata, search,
ontology, audit, cache, metrics, auth, AI-core, and Bytewax event-stream
adapters.

## What KNGR Provides

- Tenant-isolated source registration with owner, URI, evidence, confidence,
  review, and audit controls.
- Entity resolution into stable graph identities with canonical labels, types,
  aliases, attributes, source evidence, confidence, and curation status.
- Evidence-backed semantic relationships between tenant-local entities.
- Semantic enrichment records for labels and attributes produced by NLPC,
  metadata, ontology, or AI-agent workflows.
- Bounded reasoning paths with query text, relationship chains, evidence links,
  depth controls, review gates, and audit events.
- Curation decisions with curator identity, allowed decision values, evidence,
  and publication eligibility.
- Curated graph publication snapshots for generated applications and downstream
  agents.
- Provider-neutral AI knowledge-agent registration for Codex, Claude Code,
  opencode, Pi, and future runtimes.
- Bytewax-only lifecycle batch validation for source, entity, relationship,
  enrichment, reasoning, curation, publication, and knowledge-agent changes.
- Durable pending-review records for low-confidence sources, entities,
  relationships, enrichments, deep reasoning paths, and privileged knowledge
  agents, including matched rules and review reasons.
- UI route metadata and view models for source, entity, relationship,
  enrichment, reasoning, context, curation, publication, governance, audit, and
  settings screens.
- Bytewax adapter evidence for streamed knowledge mutations.
- **v2.0**: Graph traversal, subgraph extraction, BFS path finding, community
  detection, centrality scoring, entity merge/split/delete, bulk triple import,
  JSON-LD export, fact validation, provenance tracking, inference rule
  registration, concept clustering, similarity search, and analytics.

## Runtime Surfaces

- `capability_contract.py` defines configuration, deterministic rules, UI
  routes, adapters, and theme tokens.
- `service.py` is the generated-app runtime service used by tests, APIs, and
  package probes.
- `api.py` exposes dependency-light API helper functions.
- `views.py` exposes generated-app view models for KNGR screens.
- `knowledge_runtime.py` contains deterministic confidence, status, identity,
  and neighborhood helpers.
- `app.py` exposes the package semantic model and self-test.

## Lifecycle

1. Register a source with tenant, owner, URI, evidence, connector, and
   confidence.
2. Resolve entities from registered sources and attach source evidence.
3. Link relationships between tenant-local entities with predicates, source
   references, evidence links, and confidence.
4. Enrich entities with semantic labels and attributes from NLPC, ontology,
   metadata, or AI-agent workflows.
5. Build bounded reasoning paths over relationships with evidence links.
6. Inspect pending-review queues for low-confidence evidence and deep reasoning
   paths, then curate entities with explicit reviewer identity, decision, and
   evidence.
7. Publish curated graph snapshots for generated applications.
8. Register knowledge agents for source, entity, relationship, enrichment,
   reasoning, curation, publication, and lifecycle governance.
9. Validate lifecycle batches through Bytewax processor policy.
10. Inspect dashboard, context neighborhoods, governance rules, and audit
    events.

## Quick Start

```python
from capabilities.common.kngr.service import KngrService

service = KngrService()
source = service.register_source(
    source_id="src-procurement",
    tenant_id="tenant-a",
    name="Procurement events",
    source_uri="meta://procurement/events",
    owner="knowledge-steward",
    evidence_refs=["meta:source:procurement"],
    confidence_score=0.94,
    connector="meta",
)
request = service.resolve_entity(
    entity_id="entity-request",
    tenant_id="tenant-a",
    canonical_label="Purchase request 1001",
    entity_type="purchase_request",
    source_id=source["id"],
    source_evidence_refs=["doc:pr-1001"],
    aliases=["PR-1001"],
    attributes={"amount": 9500},
)
supplier = service.resolve_entity(
    entity_id="entity-supplier",
    tenant_id="tenant-a",
    canonical_label="Acme Supplies",
    entity_type="supplier",
    source_id=source["id"],
    source_evidence_refs=["doc:supplier-acme"],
)
relationship = service.link_relationship(
    relationship_id="rel-request-supplier",
    tenant_id="tenant-a",
    subject_entity_id=request["id"],
    predicate="uses_supplier",
    object_entity_id=supplier["id"],
    source_id=source["id"],
    evidence_links=["doc:pr-1001"],
    confidence_score=0.89,
)
service.curate_entity(
    curation_id="curate-request",
    tenant_id="tenant-a",
    entity_id=request["id"],
    curator="knowledge-steward",
    decision="approved",
    evidence_links=["review:curation-1"],
)
agent = service.register_knowledge_agent(
    agent_id="knowledge-steward-agent",
    tenant_id="tenant-a",
    name="Knowledge Steward Agent",
    runtime="codex",
    role="knowledge_steward",
    scope="procurement entity and relationship review",
    owner="knowledge-platform",
    purpose="review curated procurement graph quality",
)
batch = service.validate_kngr_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=1,
    operation="knowledge_agent_batch",
)
```

## Core API

| Method | Signature (key args) | Returns |
|---|---|---|
| `register_source` | `source_id, tenant_id, name, source_uri, owner, evidence_refs, confidence_score` | `dict` |
| `resolve_entity` | `entity_id, tenant_id, canonical_label, entity_type, source_id, source_evidence_refs` | `dict` |
| `link_relationship` | `relationship_id, tenant_id, subject_entity_id, predicate, object_entity_id, source_id, evidence_links, confidence_score` | `dict` |
| `enrich_entity` | `enrichment_id, tenant_id, entity_id, semantic_labels, attributes, evidence_links, confidence_score` | `dict` |
| `build_reasoning_path` | `path_id, tenant_id, query, start_entity_id, end_entity_id, relationship_ids, evidence_links` | `dict` |
| `curate_entity` | `curation_id, tenant_id, entity_id, curator, decision, evidence_links` | `dict` |
| `publish_graph` | `publication_id, tenant_id, name, entity_ids, relationship_ids, published_by, curation_recorded` | `dict` |
| `register_knowledge_agent` | `agent_id, tenant_id, name, runtime, role, scope, owner, purpose` | `dict` |
| `validate_kngr_lifecycle_batch` | `tenant_id, event_stream, mutation_count, operation` | `dict` |
| `context_neighborhood` | `tenant_id, entity_id` | `dict` |
| `dashboard_summary` | `tenant_id` | `dict` |
| `list_knowledge_graph` | `tenant_id` | `dict` |
| `describe` | `tenant_id` | `dict` |
| `evaluate` | `context: dict` | `dict` |

## v2.0 Extended API

| Method | Signature (key args) | Returns |
|---|---|---|
| `entity_add` | `entity_id, tenant_id, canonical_label, entity_type, source_id, evidence_refs` | `dict` |
| `entity_update` | `entity_id, tenant_id, properties, actor` | `dict` |
| `entity_delete` | `entity_id, tenant_id, cascade, actor` | `dict` |
| `entity_merge` | `tenant_id, primary_entity_id, secondary_entity_id, merged_by` | `dict` |
| `entity_split` | `tenant_id, source_entity_id, new_entity_id, new_label, split_attributes, split_by` | `dict` |
| `entity_search` | `tenant_id, query, entity_type, limit` | `list[dict]` |
| `similarity_entities` | `tenant_id, entity_id, limit` | `list[dict]` |
| `relation_add` | `relationship_id, tenant_id, subject_entity_id, predicate, object_entity_id, source_id, evidence_links` | `dict` |
| `graph_merge` | `source_id, target_id, tenant_id, merge_strategy, actor` | `dict` |
| `graph_traverse` | `tenant_id, start_entity_id, max_depth` | `dict` |
| `subgraph_extract` | `root_id, tenant_id, depth` | `dict` |
| `path_find` | `from_id, to_id, tenant_id, max_hops` | `dict` |
| `community_detect` | `tenant_id, algorithm` | `dict` |
| `centrality_compute` | `tenant_id, metric` | `dict` |
| `import_triples` | `triples_list, tenant_id, source_id, actor` | `dict` |
| `export_jsonld` | `tenant_id, entity_ids` | `dict` |
| `graph_export` | `tenant_id, format` | `dict` |
| `fact_validate` | `subject, predicate, object, tenant_id` | `dict` |
| `provenance_record` | `fact_id, source, confidence, tenant_id, actor` | `dict` |
| `provenance_track` | `tenant_id, entity_id` | `dict` |
| `inference_rule` | `tenant_id, rule_id, subject_type, predicate, object_type, inferred_predicate, owner` | `dict` |
| `ontology_import` | `tenant_id, ontology_id, name, entity_type_defs, predicate_defs, owner` | `dict` |
| `conflict_detect` | `tenant_id` | `list[dict]` |
| `concept_cluster` | `tenant_id, entity_type` | `dict[str, list[str]]` |
| `concept_similarity_matrix` | `concept_ids, tenant_id` | `dict` |
| `kg_analytics` | `period, tenant_id` | `dict` |
| `knowledge_graph_health` | `tenant_id` | `dict` |

## World-Class Enhancements (v2.0)

The following 15 improvements bring KNGR to production-grade quality, closing
the gap with commercial graph databases and ML-powered knowledge platforms.

| # | Name | Category | Summary |
|---|---|---|---|
| I1 | Persistent Graph Storage | Infrastructure | Replace in-memory dicts with async SQLAlchemy + PostgreSQL (`asyncpg`). Upsert writes, cursor-paginated reads, ACID guarantees. No state loss on restart. |
| I2 | Vector-Similarity Semantic Search | Query / Intelligence | `semantic_search(query, tenant_id, top_k, threshold)` via Ollama `nomic-embed-text` + `pgvector` cosine similarity (`<=>`). Sub-50ms p99 at scale. |
| I3 | Async-First Service Layer | Performance / Architecture | All public methods converted to `async def`. `asyncio.gather()` fan-out in `list_knowledge_graph`. Natively awaitable by FastAPI / Starlette. |
| I4 | Transitive Inference Engine | Reasoning | `apply_inference_rules(tenant_id)` materialises implicit facts via registered rule patterns. Confidence decay: `min(r1, r2) * 0.9` per hop. Inferred flag on `KnowledgeRelationship`. |
| I5 | Multi-Hop SPARQL-Subset Query API | Query Language | `sparql_query(sparql_text, tenant_id)` parses restricted `SELECT ?x WHERE { ... }` patterns via `rdflib`. Enables ad-hoc graph analysis without Python code. |
| I6 | Confidence Decay Over Time | Knowledge Quality | `refresh_confidence(tenant_id, half_life_days)` applies `0.5 ** (age_days / half_life)` decay. Entities below threshold transition to `pending_review`. |
| I7 | Change-Data-Capture Event Streaming | Integration | `stream_mutations(tenant_id, since_cursor)` async generator via PostgreSQL `LISTEN/NOTIFY`. Zero-lag CDC for search indexers, alert engines, and warehouses. |
| I8 | Role-Based Access Control per Entity Type | Security | `check_entity_permission(actor_roles, entity_type, operation, tenant_id)` enforces column/row-level security. Blocks cross-BU PII leakage. |
| I9 | Ontology-Driven Entity Type Validation | Data Quality | `validate_entity_type(entity_type, tenant_id)` rejects unregistered types unless `strict_ontology=False`. Eliminates `"PurchaseRequest"` / `"purchase_request"` / `"PR"` fragmentation. |
| I10 | Incremental Subgraph Embedding for GraphRAG | AI Integration | `embed_subgraph(root_id, tenant_id, depth, model)` stores `pgvector` embeddings for Ollama `nomic-embed-text`. Dirty-flag queue re-embeds only mutated subgraphs. |
| I11 | Distributed Graph Sharding | Scalability | `shard_graph(tenant_id, shard_count)` assigns entities by `hash(entity_id) % shard_count`. Cross-shard queries via `asyncio.gather()` fan-out + merge. |
| I12 | Graph Versioning and Temporal Snapshots | Audit / Compliance | `graph_snapshot` + `graph_at(tenant_id, timestamp)` via event-sourced `kngr_deltas` table. Supports SOX / GDPR Article 17 time-travel reconstruction. |
| I13 | Uncertainty-Aware Probabilistic Reasoning | Reasoning / AI | `probabilistic_infer(query, tenant_id, samples)` models confidence as `Beta(alpha, beta)` and propagates via Monte Carlo sampling. Returns `mean`, `std`, `ci_95`. |
| I14 | Automated Entity Deduplication | Data Quality / ML | `auto_deduplicate(tenant_id, threshold, dry_run)` uses blocking keys + Jaro-Winkler similarity to surface candidate merges. Auto-merges via `graph_merge` when `dry_run=False`. |
| I15 | Federated Knowledge Graph Query | Enterprise Composition | `federated_query(query, tenant_ids, actor_roles)` fans out across authorised tenants, merges via I14 dedup, annotates `source_tenant` per entity, emits per-tenant audit events. |

## New Methods — Usage Examples

### 1. BFS Path Finding

Find the shortest relationship chain between two entities:

```python
result = service.path_find(
    from_id="entity-request",
    to_id="entity-supplier",
    tenant_id="tenant-a",
    max_hops=5,
)
# {"from_id": "entity-request", "to_id": "entity-supplier",
#  "path": ["entity-request", "entity-supplier"], "found": True, "hops": 1}
```

### 2. Subgraph Extraction

Extract a 2-hop neighbourhood around a root entity for GraphRAG context:

```python
subgraph = service.subgraph_extract(
    root_id="entity-request",
    tenant_id="tenant-a",
    depth=2,
)
# {"root_id": ..., "depth": 2, "node_count": 4, "edge_count": 3,
#  "nodes": [...], "edges": [...]}
```

### 3. Community Detection

Identify densely connected clusters (connected-component proxy for Louvain):

```python
communities = service.community_detect(tenant_id="tenant-a", algorithm="louvain")
# {"algorithm": "louvain", "community_count": 2,
#  "communities": [{"id": 0, "members": [...]}, ...]}
```

### 4. Bulk Triple Import

Ingest RDF-style triples without calling `resolve_entity` / `link_relationship`
individually:

```python
result = service.import_triples(
    triples_list=[
        {"subject": "entity-a", "predicate": "related_to", "object": "entity-b"},
        {"subject": "entity-b", "predicate": "part_of",    "object": "entity-c"},
    ],
    tenant_id="tenant-a",
    source_id="src-procurement",
)
# {"triples_submitted": 2, "entities_created": 3, "relationships_created": 2}
```

### 5. Provenance Tracking

Retrieve the complete source + audit trail for a single entity:

```python
prov = service.provenance_track(tenant_id="tenant-a", entity_id="entity-request")
# {"entity_id": "entity-request", "source": {...}, "audit_trail": [...events...]}
```

### 6. Entity Merge

Collapse a duplicate entity, re-pointing all relationships to the canonical record:

```python
result = service.entity_merge(
    tenant_id="tenant-a",
    primary_entity_id="entity-supplier",
    secondary_entity_id="entity-supplier-duplicate",
    merged_by="knowledge-steward",
)
# Primary gains secondary's aliases; secondary is marked "retired".
```

### 7. JSON-LD Export

Export graph as a JSON-LD document for downstream semantic-web consumers:

```python
doc = service.export_jsonld(tenant_id="tenant-a")
# {"@context": "https://schema.org/", "@graph": [...], "relationships": [...]}
```

## Guardrails

KNGR denies operations without tenant context, source identity, source owner,
source URI, evidence, positive confidence, entity identity, labels, types,
relationship endpoints, predicates, reasoning queries, curation decisions,
publication names, publishers, or curated publication entities. Review-required
operations are durable: low-confidence source, entity, relationship, and
enrichment records, as well as deep reasoning paths and privileged
knowledge-agent registrations, are stored with `pending_review` status, matched
rule names, and review reasons. Batch knowledge mutations must use Bytewax.
Cross-tenant access and unaudited graph state changes are blocked. KNGR denies
knowledge-agent registrations that use unsupported runtimes or roles, omit
scope, owner, or purpose, or hide machine contribution. Lifecycle batches that
are not routed through Bytewax are denied and retained as denied batch evidence.

## Composition

KNGR depends on GRPH, NLPC, META, SRCH, ONTO, AICR, and CONF for graph
structure, semantic processing, metadata, discovery, vocabulary, AI-agent, and
configuration context. Optional adapters connect it to AUTH, AUDL, MONI, CACH,
and Bytewax-backed event streams. Generated applications compose KNGR through
the semantic model, UI manifest, API helpers, first-class agent manifest,
streaming manifest, service runtime, rule engine, and theme contract.

Planned v2.0 integrations (I1–I15) additionally depend on `asyncpg`, `pgvector`,
Ollama `nomic-embed-text`, `rdflib`, and PostgreSQL `LISTEN/NOTIFY` for
persistence, vector search, declarative querying, CDC streaming, and temporal
snapshots.
