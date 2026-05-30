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
8. Expose UI route metadata, view models, theme tokens, audit events, and package evidence.

The capability does not require a live graph database, vector store, Bytewax worker, LLM provider, or browser runtime for the generated-app baseline. Those are adapters.

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
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

The `observability.event_stream` and `adapters.event_stream` values are `bytewax`.

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

## UI Requirements

The generated UI manifest exposes 12 route surfaces:

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
- Audit
- Settings

Theme components cover hybrid results, graph source cards, vector index cards, reasoning paths, provenance panels, generation panels, curation queues, audit timelines, and query console displays.

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
- Generic APG record compatibility
- Package/dashboard summaries

## Adapter Boundaries

Generated apps should call `grag_runtime.GragService`. Production deployments may replace or wrap it with `service.GraphRAGService` and concrete adapters for graph storage, vector indexes, model routing, policy, auth, audit, metrics, cache, and Bytewax streams.

## Acceptance Criteria

- The capability has root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
- The contract exposes at least 30 rules, 12 UI routes, Bytewax adapter configuration, and visual theme metadata.
- The runtime executes the lifecycle from source registration to publication.
- Guardrails block missing tenant context, source data, indexes, evidence, provenance, citations, curation evidence, unsafe answers, and unapproved publication.
- The API and app entrypoint import without production dependencies.
- Package semantic evidence is generated from the current contract.
- Focused tests cover the lifecycle, guardrails, view helpers, package shape, and import-light API.
