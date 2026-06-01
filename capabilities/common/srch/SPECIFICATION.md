# SRCH Specification

## Purpose

SRCH provides APG with first-class enterprise search for generated
applications. It supports governed indexing, document ingestion, bulk indexing,
keyword search, semantic search, hybrid search, faceted navigation, access
filtering, query analytics, first-class AI search-agent composition, Bytewax
lifecycle batch validation, UI composition, and auditable policy enforcement.

## Scope

This packet establishes the executable baseline for SRCH:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- Provider-neutral AI search agents as executable state with runtime, role,
  scope, owner, purpose, disclosure, human-review status, and audit evidence.
- Bytewax-only lifecycle batch validation for search mutations.
- Durable review-required index, document, and query outcomes as
  pending-review records with matched rules and review reasons.
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
- AI search agent: assists with source curation, index review, document
  quality, query relevance, ranking, access policy, facet taxonomy, and
  lifecycle batch review while remaining provider-neutral.
- Platform operator: configures Bytewax streams, ETLP, META, NLPC, AICR, auth,
  audit, cache, metrics, and generated app deployment.

## Functional Requirements

### Index Lifecycle

- Create indices with tenant, name, owner, content type, classification, source
  lineage, embedding readiness, and status.
- Deny missing tenant, name, owner, content type, or classification.
- Deny restricted indices without lineage.
- Persist unknown content types and classifications as pending-review indices
  with deterministic matched-rule and reason evidence.

### Document Lifecycle

- Index documents with tenant, index, document id, title, body, classification,
  facets, metadata, and source lineage.
- Deny missing index, id, title, body, classification, or lineage.
- Persist facet keys outside the configured allowlist as pending-review
  documents with deterministic matched-rule and reason evidence.

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
- Persist large result windows and unknown query types as review-required query
  records.
- Record query status, decision, matched rules, review reasons, required
  actions, result count, and audit events.

### AI Agent Lifecycle

- Register search agents with tenant, name, runtime, role, scope, owner,
  purpose, machine-contribution disclosure, and human-approval metadata.
- Support provider-neutral runtimes `codex`, `claude_code`, `opencode`, and
  `pi`.
- Support roles for source curation, index review, document-quality review,
  query-relevance review, ranking review, access-policy review,
  facet-taxonomy review, lifecycle-batch review, and search stewardship.
- Deny unsupported runtimes, unsupported roles, missing scope, missing owner,
  missing purpose, and missing machine-contribution disclosure.
- Put privileged search-agent roles into pending review when human approval
  evidence is absent.
- Keep live agent invocation, credentials, and provider-specific routing behind
  the AICR adapter boundary.

### Bytewax Lifecycle Batches

- Validate SRCH lifecycle mutation batches through the declared Bytewax stream
  contract.
- Accept only configured lifecycle operations: index, document, bulk indexing,
  query, facet, ranking, access-policy, and search-agent batches.
- Deny non-Bytewax lifecycle streams while preserving denied-batch evidence for
  audit and UI review.

### UI and Theming

- Expose routes for dashboard, search, indices, documents, bulk indexing,
  facets, analytics, ranking, access review, governance, agents, lifecycle
  batches, audit, and settings.
- Provide route-specific view models.
- Expose pending-review queues for index, document, query, and search-agent
  governance.
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
- Live AI-agent CLI/API invocation.
- Persistent database migrations.
- Browser-rendered UI validation.
- Load, latency, recall, ranking, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 39 deterministic rules, at least
  14 UI routes, first-class agent metadata, Bytewax lifecycle metadata, runtime
  adapter evidence, and theme component metadata.
- `SrchService` executes index, document, bulk, query, facet, list, dashboard,
  search-agent, lifecycle-batch, audit, and APG record compatibility flows.
- Guardrail tests prove denied cases fail before invalid state is accepted, and
  review-required cases persist as pending-review records with matched rules and
  review reasons.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
