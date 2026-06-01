# GRAG Capability Plan

## 1. Specification

- Define the generated-app lifecycle for graph sources, vector sources, hybrid retrieval, reasoning, generation, curation, publication, and audit.
- Extend the generated-app lifecycle with provider-neutral GraphRAG agents and
  Bytewax lifecycle batches.
- Separate executable generated-app behavior from production adapters.
- Require Bytewax for batch/event stream adapter declarations.

## 2. Contract

- Expand configuration to cover source management, retrieval, reasoning, generation, provenance, curation, security, agents, streaming, governance, observability, adapters, UI, and theme.
- Expand deterministic rules beyond 45 guardrails.
- Add GraphRAG-agent guardrails for supported runtime, role, scope, owner,
  purpose, contribution disclosure, and privileged-role approval.
- Add Bytewax lifecycle stream metadata for GraphRAG lifecycle batches.
- Expose 14 UI routes and component theme hooks.
- Add comparison operators to rule matching for confidence and count thresholds.

## 3. Runtime

- Add `grag_runtime.py` with `GragRecord` and `GragService`.
- Implement source registration, hybrid query, reasoning path, answer generation, curation, publication, audit, dashboard, package, and listing methods.
- Implement provider-neutral GraphRAG-agent records, lifecycle batch records,
  registration, validation, listing, dashboard summaries, and audit events.
- Enforce deny results in runtime methods and persist review-required results
  as pending-review records with policy evidence.

## 4. API And UI Helpers

- Replace heavy API entrypoint with import-light helper functions that wrap `GragService`.
- Add generated-app UI helper models to `views.py` while preserving legacy
  model exports for heavier production modules, including agent roster and
  lifecycle batch monitor models.
- Add pending-review queues to dashboard, source, retrieval, reasoning,
  generation, curation, governance, API, and package surfaces.

## 5. Package Evidence

- Replace static semantic model data with contract-derived metadata in `app.py`.
- Refresh `semantic_model.json`, `package_manifest.json`, and `release_report.json`.

## 6. Focused Verification

- Compile the changed Python files.
- Run the GRAG contract and package tests only.
- Run `app.self_test()`.
- Run APG implementation audit and publish-plan checks for GRAG.
- Scan the primary slice for stale markers and exaggerated language.
- Run whitespace diff checks on GRAG and the progress log.

## 7. Review And Commit

- Perform a direct review of the changed packet.
- Fix issues found during focused verification.
- Update `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.
