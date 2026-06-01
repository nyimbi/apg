# RAGN Capability Specification

## Purpose

RAGN gives APG applications a governed retrieval-augmented generation surface.
It converts tenant-approved documents and context into cited answers while
preserving source attribution, access filtering, safety checks, review gates,
conversation traces, and audit evidence.

## Functional Scope

RAGN must provide:

- Knowledge-base lifecycle records with id, tenant, name, owner, source
  attribution, classification, status, and audit events.
- Document ingestion records with knowledge base, title, source URI, content
  hash, classification, batch-size review policy, and audit events.
- Retrieval records with query, knowledge base, document evidence, result
  window, source classification, access-filter state, confidence score, review
  state, and audit events.
- Answer generation records with retrieval context, query, answer text,
  citations, model location, model policy, prompt-injection checks,
  unsafe-answer checks, pending-context review state, and audit events.
- Conversation-turn records with conversation id, user id, answer linkage,
  turn count, review state, and audit events.
- Citation validation requiring source, document, and chunk identifiers.
- Answer curation with curator, decision, evidence, status, and audit events.
- First-class RAG-agent composition for Codex, Claude Code, opencode, and Pi
  style runtimes, with supported RAG roles, owner, purpose, bounded scope,
  machine-contribution disclosure, privileged-role approval status, and audit
  events.
- Bytewax lifecycle batch validation for corpus, document, retrieval, context,
  generation, citation, evaluation, safety, and RAG-agent batch operations.
- Dashboard summaries, aggregate package listings, route metadata, and
  generated-app view models.
- Deterministic rule evaluation for lifecycle and safety guardrails.
- Durable review-evidence fields for review-required records: decision,
  matched rules, review reasons, and audit evidence.
- Theme tokens and named UI components for generated RAGN screens.
- Package evidence through `semantic_model.json`, `package_manifest.json`, and
  `release_report.json`.

## Configuration Contract

The capability configuration is tenant-scoped and contains these sections:

- `knowledge_bases`: id, name, owner, source attribution, classification, and
  retirement policy.
- `documents`: title, content hash, knowledge-base linkage, classification,
  batch-size, and ingestion review policy.
- `chunking`: chunk-size and overlap bounds.
- `retrieval`: semantic retrieval, keyword fallback, confidence, result window,
  and RBAC filtering policy.
- `generation`: citation, model policy, streaming, external model, and answer
  length policy.
- `conversations`: conversation id, user id, memory, turn count, and retention
  policy.
- `citations`: source, document, chunk, and minimum citation policy.
- `curation`: low-confidence review, curator, evidence, and decision policy.
- `security`: tenant isolation, restricted-source filters, prompt-injection
  scan, and unsafe answer blocking.
- `agents`: first-class provider-neutral RAG-agent runtime, role, owner,
  purpose, scope, contribution-disclosure, and privileged-role approval policy.
- `streaming`: Bytewax lifecycle stream, event-time watermark, required
  processor, allowed batch operations, stream topics, and broker-dependency
  policy.
- `governance`: tenant context and audit requirements.
- `observability`: metrics, trace, audit, and Bytewax event-stream policy.
- `adapters`: generated runtime, production runtime, HTTP API, Bytewax, SRCH,
  NLPC, AICR, MLCM, KNGR, GRPH, META, AUTH, AUDL, CACH, and MONI integration
  points.
- `ui`: feature toggles for generated screens.
- `theme`: named APG visual theme and tenant override policy.

## Rule Engine

The rule engine is deterministic and evaluates plain dictionaries. It returns
`allow`, `require_review`, or `deny` with matched rule names and required
actions. Rules cover:

- Tenant context.
- Knowledge-base identity, name, owner, and source attribution.
- Document knowledge base, title, content hash, source URI, classification, and
  large-ingest review.
- Chunk-size bounds.
- Retrieval query, knowledge base, result window, restricted-source filter, and
  low-context-confidence review.
- Generation query, context, citations, external model policy,
  prompt-injection blocking, unsafe-answer blocking, and pending-context
  generation review.
- Conversation id, user id, and long-conversation review.
- Citation source, document, and chunk identifiers.
- Curation curator, decision, and evidence.
- Bytewax requirement for batch mutation flows.
- Cross-tenant denial and audit requirement for state changes.
- RAG-agent supported runtime, supported role, explicit scope, accountable
  owner, declared purpose, machine-contribution disclosure, and human approval
  review status for privileged roles.
- Bytewax-only RAGN lifecycle batch validation.

## UI Contract

RAGN exposes route metadata and view-model helpers for:

- Dashboard
- Studio
- Knowledge bases
- Documents
- Retrieval
- Generation
- Conversations
- Citations
- Curation
- Governance
- Agents
- Lifecycle batch monitor
- Audit
- Settings

Generated UIs should prioritize dense operational screens: source and answer
evidence, confidence, review queues, retrieval diagnostics, citation stacks,
conversation traces, governance rules, and audit timelines.

Review-required outcomes must remain inspectable in those screens. Large
ingests, large retrieval windows, low-confidence retrievals, answers generated
from pending-review context, long conversations, and privileged RAG-agent
registrations are stored with `pending_review` status and policy evidence.
Deny outcomes remain hard-blocking and should be reported as policy failures.

## Adapter Boundaries

The dependency-light runtime stores lifecycle records in memory for
generated-app package tests and local composition. Production persistence,
vector search, hybrid retrieval, generation models, streaming, auth, audit
persistence, metrics, cache, and event processing remain adapter
responsibilities. Batch mutation flows use Bytewax as the event-stream engine.

## Non-Goals

This packet does not implement a persistent vector index, live model inference,
database migrations, browser-rendered UI, or production service orchestration.
It defines the executable APG capability surface and adapter seams that those
systems attach to.
