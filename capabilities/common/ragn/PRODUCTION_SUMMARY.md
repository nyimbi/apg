# RAGN Production Summary

RAGN currently has a verified generated-app capability packet and a separate
production adapter surface. The generated-app surface is import-light and
dependency-light; the production service stack still requires live database,
retrieval, model, and observability adapters before it can be called production
ready.

## Verified In This Packet

- Executable `capability_contract.py` with lifecycle configuration, deterministic
  guardrails, first-class RAG agents, Bytewax lifecycle batches, UI routes,
  adapters, and theme components.
- Import-light `rag_runtime.RagnService` for generated applications.
- API helpers and UI view models that do not import the heavier async database
  stack.
- Contract-derived `app.py`, `semantic_model.json`, `release_report.json`, and
  `package_manifest.json`.
- Focused pytest coverage for lifecycle, guardrails, package contract, and
  import-light API behavior.

## Adapter Work Still Required

- Persistent storage and migrations.
- SRCH/NLPC retrieval wiring.
- AICR/MLCM model policy wiring.
- AUTH/AUDL/MONI/CACH adapter integration.
- Durable Bytewax topologies for ingestion, re-indexing, agent events, and
  lifecycle batches.
- External Codex, Claude Code, opencode, and Pi adapter clients behind the
  provider-neutral RAG-agent contract.
- Rendered browser UI verification.
- Performance, resilience, and retention-policy validation.
