# KNGR Capability Specification

## Purpose

KNGR turns APG application facts into a tenant-scoped knowledge graph that can
be queried, reviewed, published, and composed into generated applications. It is
the semantic layer that lets ERP components, AI agents, search, GraphRAG, and
workflow tools share explainable business context without losing provenance.

## Functional Scope

KNGR must provide:

- Source registration with id, tenant, display name, URI, owner, connector,
  evidence references, confidence score, status, review state, and audit event.
- Entity resolution with stable entity id, canonical label, entity type,
  registered source, source evidence, aliases, attributes, confidence, and
  curation status.
- Relationship linking between tenant-local entities with predicate, source,
  evidence links, confidence score, review state, status, and audit event.
- Semantic enrichment with labels, attributes, evidence links, confidence,
  review state, and lifecycle status.
- Bounded reasoning paths with query, start/end entities, relationship chain,
  evidence links, depth, review state, status, and audit event.
- Curation decisions with curator, allowed decision, evidence, notes, and entity
  curation-state updates.
- Publication snapshots for curated entity and relationship sets.
- Dashboard summaries, graph listings, context neighborhoods, route metadata,
  and generated-app view models.
- Deterministic rule evaluation for all high-risk lifecycle operations.
- Theme tokens and named components for generated APG UI surfaces.
- Package evidence through `semantic_model.json`, `package_manifest.json`, and
  `release_report.json`.

## Configuration Contract

The capability configuration is tenant-scoped and contains these sections:

- `sources`: source identity, owner, evidence, confidence, and review policy.
- `entities`: entity identity, label, type, source, evidence, and publication
  curation policy.
- `relationships`: subject/object/predicate/source/evidence/confidence policy.
- `enrichment`: semantic label, evidence, confidence, review, and NLPC adapter
  policy.
- `reasoning`: bounded depth, query, endpoint, evidence, and review policy.
- `curation`: curator, allowed decisions, and evidence policy.
- `publication`: publication name, publisher, curation, and entity-count policy.
- `security`: tenant isolation and public graph restrictions.
- `governance`: audit requirements for source, entity, relationship,
  enrichment, reasoning, and publication state changes.
- `observability`: metrics, trace, audit, and Bytewax event-stream policy.
- `adapters`: generated runtime, helper runtime, HTTP API, Bytewax, GRPH, NLPC,
  META, SRCH, ONTO, AICR, AUTH, AUDL, CACH, and MONI integration points.
- `ui`: feature toggles for generated screens.
- `theme`: named APG visual theme and tenant override policy.

## Rule Engine

The rule engine is deterministic and evaluates plain dictionaries. It returns
`allow`, `require_review`, or `deny` with matched rule names and actions. Rules
cover:

- Tenant context.
- Source identity, name, URI, owner, evidence, confidence, and review.
- Entity identity, canonical label, type, source, source evidence, confidence,
  and review.
- Relationship subject, object, predicate, source, evidence, confidence, and
  review.
- Enrichment labels, evidence, confidence, and review.
- Reasoning query, entity endpoints, evidence, depth, and review.
- Curation curator, decision, allowed value, and evidence.
- Publication name, publisher, curation, and entity count.
- Bytewax requirement for batch knowledge mutations.
- Cross-tenant denial and audit requirement for state changes.

## UI Contract

KNGR exposes route metadata and view-model helpers for:

- Dashboard
- Sources
- Entities
- Relationships
- Enrichment
- Reasoning
- Context
- Curation
- Publication
- Governance
- Audit
- Settings

Generated UIs should present dense operational screens: list/detail panels,
review queues, relationship context, evidence trails, and audit timelines. The
theme contract defines source panels, entity cards, relationship panels,
semantic graph visuals, enrichment panels, reasoning paths, curation queues,
publication cards, context panels, and audit timelines.

## Adapter Boundaries

The dependency-light runtime stores records in memory for generated-app package
tests and local composition. Production persistence, graph storage, NLP
extraction, ontology mapping, metadata sync, authorization, audit persistence,
metrics, cache, and event-stream execution are adapter responsibilities exposed
through the contract. Batch mutation flows use Bytewax as the event-stream
engine.

## Non-Goals

This packet does not implement a persistent graph database, vector index,
ontology editor, live Bytewax topology, browser-rendered UI, or external AI
agent provider integration. It defines the executable APG capability surface and
adapter seams that those systems attach to.
