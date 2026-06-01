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

## Example

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
