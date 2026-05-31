# GRAG Capability Specification

## Purpose

GRAG turns APG knowledge assets into explainable graph-grounded answers. It should let an application compose graph sources, vector indexes, hybrid retrieval, multi-hop reasoning, provenance, answer generation, curation, publication, UI screens, and deterministic guardrails without first wiring a production graph database or model provider.

## Scope

GRAG owns the generated-app lifecycle for graph-based RAG:

1. Register graph sources with tenant, owner, graph id, classification, and provenance.
2. Register vector sources with tenant, index id, embedding model, source documents, and owner.
3. Run hybrid retrieval against one graph source and one vector source.
4. Build a reasoning path from the hybrid query through graph evidence.
5. Generate an answer grounded in retrieval context and reasoning path evidence.
6. Curate the answer with an allowed review decision and evidence.
7. Publish only approved answers.
8. Register provider-neutral GraphRAG agents with bounded graph, retrieval,
   reasoning, provenance, generation, citation, safety, lifecycle, and steward
   roles.
9. Validate GraphRAG lifecycle batches through Bytewax-first processor
   contracts.
10. Expose UI route metadata, view models, theme tokens, audit events, and package evidence.

The capability does not require a live graph database, vector store, Bytewax
worker, LLM provider, or browser runtime for the dependency-light generated-app
surface. Those are adapters.

## Runtime Model

`GragService` stores lifecycle records in memory for generated apps and tests:

- `graph_source`
- `vector_source`
- `hybrid_query`
- `reasoning_path`
- `answer`
- `curation`
- `publication`
- `audit_event`

Each record includes id, tenant id, kind, status, metadata, and creation timestamp. Tenant isolation is enforced on every lookup.

## Configuration

The contract has explicit sections for:

- `graph_sources`
- `vector_sources`
- `hybrid_retrieval`
- `reasoning`
- `generation`
- `provenance`
- `curation`
- `security`
- `agents`
- `streaming`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

The `observability.event_stream` and `adapters.event_stream` values are `bytewax`.
The `streaming.required_processor` value is `bytewax`, and the lifecycle stream
is `grag.lifecycle`.

## Rule Engine

The rule engine is deterministic. Rules are evaluated against context dictionaries and may return:

- `allow`
- `require_review`
- `deny`

Runtime methods raise `PermissionError` for denied operations and for review-required operations where review evidence is missing.

Rule categories:

- Tenant context and tenant isolation.
- Graph-source registration and retirement review.
- Vector-source readiness.
- Hybrid retrieval query, source, index, access-filter, result-window, and confidence guardrails.
- Reasoning start-node, hop-count, evidence-path, explanation, and multi-hop review guardrails.
- Generation query, retrieval context, reasoning path, answer text, provenance, citation, model policy, unsafe answer, and confidence guardrails.
- Curation and publication approval guardrails.
- Bytewax event-stream and audit guardrails.
- GraphRAG-agent supported runtime, supported role, explicit scope, accountable
  owner, declared purpose, machine-contribution disclosure, and human approval
  review status for privileged roles.
- Bytewax-only GRAG lifecycle batch validation.

## UI Requirements

The generated UI manifest exposes 14 route surfaces:

- Dashboard
- Query
- Graph sources
- Vector sources
- Hybrid retrieval
- Reasoning
- Provenance
- Generation
- Curation
- Governance
- Agents
- Lifecycle batch monitor
- Audit
- Settings

Theme components cover hybrid results, graph source cards, vector index cards, reasoning paths, provenance panels, generation panels, curation queues, audit timelines, and query console displays.
Theme components also include GraphRAG-agent roster and Bytewax lifecycle panel
hooks.

## API Requirements

`api.py` exposes import-light helper functions for:

- Capability status
- Graph-source registration and retirement
- Vector-source registration
- Hybrid query execution
- Reasoning path construction
- Answer generation
- Curation
- Publication
- GraphRAG-agent registration and listing
- Lifecycle batch validation and listing
- Generic APG record compatibility
- Package/dashboard summaries

## Adapter Boundaries

Generated apps should call `grag_runtime.GragService`. Production deployments may replace or wrap it with `service.GraphRAGService` and concrete adapters for graph storage, vector indexes, model routing, policy, auth, audit, metrics, cache, and Bytewax streams.

## Acceptance Criteria

- The capability has root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
- The contract exposes at least 45 rules, 14 UI routes, first-class agents,
  Bytewax lifecycle streaming, Bytewax adapter configuration, and visual theme
  metadata.
- The runtime executes the lifecycle from source registration to publication.
- The runtime executes GraphRAG-agent registration and Bytewax lifecycle batch
  validation.
- Guardrails block missing tenant context, source data, indexes, evidence, provenance, citations, curation evidence, unsafe answers, and unapproved publication.
- Guardrails block unsupported agent runtimes, unsupported roles, missing
  scope, missing owner, missing purpose, missing contribution disclosure, and
  non-Bytewax lifecycle streams.
- The API and app entrypoint import without production dependencies.
- Package semantic evidence is generated from the current contract.
- Focused tests cover the lifecycle, guardrails, view helpers, package shape, and import-light API.
