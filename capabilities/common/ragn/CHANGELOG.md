# RAGN Changelog

## 1.0.0 - Generated-App Capability Packet

- Added current capability contract sections for knowledge bases, documents,
  chunking, retrieval, generation, conversations, citations, curation, security,
  governance, observability, adapters, UI, and theme.
- Added deterministic guardrails for tenant context, source attribution,
  document evidence, restricted-source filtering, context confidence, citations,
  model policy, prompt-injection blocking, unsafe-answer blocking, conversation
  state, curation, Bytewax batch mutation, tenant isolation, and audit evidence.
- Added first-class RAG-agent composition for Codex, Claude Code, opencode, and
  Pi style runtimes with supported roles, owner, scope, purpose, contribution
  disclosure, privileged-role approval status, route metadata, and theme
  components.
- Added Bytewax lifecycle stream contracts and executable lifecycle batch
  validation for corpus, document, retrieval, context, generation, citation,
  evaluation, safety, and RAG-agent batches.
- Added `rag_runtime.RagnService` as the dependency-light generated-app runtime.
- Replaced generated-app API helpers and UI view models with import-light
  surfaces.
- Replaced static package metadata with contract-derived semantic model and
  release evidence.
- Added root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.

## Remaining Work

- Production adapter wiring remains separate from the generated-app runtime.
- Live retrieval, generation, persistence, durable Bytewax topologies, external
  agent clients, and browser UI checks are not part of this battery-conscious
  verification slice.
