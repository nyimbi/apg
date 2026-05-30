# SRCH Specification

## Purpose

SRCH provides APG with first-class enterprise search for generated
applications. It supports governed indexing, document ingestion, bulk indexing,
keyword search, semantic search, hybrid search, faceted navigation, access
filtering, query analytics, UI composition, and auditable policy enforcement.

## Scope

This packet establishes the executable baseline for SRCH:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- Package evidence that can be published and self-tested from the current
  executable contract.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

## Actors

- Content owner: creates indices and supplies source lineage.
- Indexing operator: ingests documents and manages bulk indexing.
- Search user: runs governed keyword, semantic, and hybrid queries.
- Governance reviewer: reviews restricted search, large result windows, facets,
  and denied query evidence.
- Platform operator: configures Bytewax streams, ETLP, META, NLPC, AICR, auth,
  audit, cache, metrics, and generated app deployment.

## Functional Requirements

### Index Lifecycle

- Create indices with tenant, name, owner, content type, classification, source
  lineage, embedding readiness, and status.
- Deny missing tenant, name, owner, content type, or classification.
- Deny restricted indices without lineage.
- Require review for unknown content types and classifications.

### Document Lifecycle

- Index documents with tenant, index, document id, title, body, classification,
  facets, metadata, and source lineage.
- Deny missing index, id, title, body, classification, or lineage.
- Require review for facet keys outside the configured allowlist.

### Bulk Index Lifecycle

- Bulk-index document batches into tenant-scoped indices.
- Deny empty batches and missing source lineage.
- Require review for batches beyond the configured batch size.
- Use Bytewax as the event-stream adapter for batch indexing metadata.

### Query Lifecycle

- Query one or more tenant-scoped indices using keyword, semantic, or hybrid
  query types.
- Deny missing query text, selected indices, query type, or positive result
  window.
- Deny restricted-content queries without RBAC filtering.
- Deny semantic and hybrid queries until all selected indices have embedding
  indexes ready.
- Require review for large result windows and unknown query types.
- Record query status, matched rules, required actions, result count, and audit
  events.

### UI and Theming

- Expose routes for dashboard, search, indices, documents, bulk indexing,
  facets, analytics, ranking, access review, governance, audit, and settings.
- Provide route-specific view models.
- Publish discovery-console theme tokens and component hints.

### Adapters

- Use Bytewax for batch indexing/event streams.
- Expose adapter keys for generated runtime, helper runtime, HTTP API, ETLP,
  META, NLPC, AICR, AUTH, AUDL, CACH, MONI, and vector indexing.

## Non-Goals

- Live OpenSearch, Elasticsearch, Solr, PostgreSQL FTS, or vector database
  integration.
- Live Bytewax stream execution.
- Live embedding provider calls.
- Persistent database migrations.
- Browser-rendered UI validation.
- Load, latency, recall, ranking, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 30 deterministic rules, at least
  12 UI routes, Bytewax adapter evidence, runtime adapter evidence, and theme
  component metadata.
- `SrchService` executes index, document, bulk, query, facet, list, dashboard,
  audit, and APG record compatibility flows.
- Guardrail tests prove denied or review-required cases fail before invalid
  state is accepted.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
