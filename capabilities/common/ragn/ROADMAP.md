# RAGN Roadmap

This roadmap tracks practical extensions beyond the current generated-app
capability packet. The current verified packet provides contract, rule engine,
dependency-light runtime, API helpers, UI models, package evidence, and focused
tests.

## Current Baseline

- Tenant-scoped knowledge-base lifecycle.
- Document ingestion records with content hash, source URI, classification, and
  audit evidence.
- Governed retrieval records with context confidence and restricted-source
  filtering.
- Cited answer generation records with model-policy and safety guardrails.
- Conversation-turn records and answer curation.
- Contract-derived semantic model, release report, and package manifest.

## Near-Term Work

- Wire generated-app RAGN records to the APG persistence layer.
- Connect retrieval to SRCH and NLPC adapters.
- Add model-selection policy integration with AICR and MLCM.
- Add persisted citation records and source preview rendering.
- Add Bytewax topology examples for batch document ingestion and re-indexing.
- Add browser-rendered UI screens from the current view-model contract.

## Later Work

- Add vector-index and hybrid-search adapters.
- Add KNGR/GRPH context expansion for GraphRAG handoff.
- Add offline evaluation harnesses for citation coverage, answer faithfulness,
  retrieval quality, and prompt-injection policy behavior.
- Add production deployment checks for latency, resilience, and data-retention
  policy.
