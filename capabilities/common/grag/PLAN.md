# GRAG Capability Plan

## 1. Specification

- Define the generated-app lifecycle for graph sources, vector sources, hybrid retrieval, reasoning, generation, curation, publication, and audit.
- Separate executable generated-app behavior from production adapters.
- Require Bytewax for batch/event stream adapter declarations.

## 2. Contract

- Expand configuration to cover source management, retrieval, reasoning, generation, provenance, curation, security, governance, observability, adapters, UI, and theme.
- Expand deterministic rules beyond 30 guardrails.
- Expose 12 UI routes and component theme hooks.
- Add comparison operators to rule matching for confidence and count thresholds.

## 3. Runtime

- Add `grag_runtime.py` with `GragRecord` and `GragService`.
- Implement source registration, hybrid query, reasoning path, answer generation, curation, publication, audit, dashboard, package, and listing methods.
- Enforce deny and review-required results in runtime methods.

## 4. API And UI Helpers

- Replace heavy API entrypoint with import-light helper functions that wrap `GragService`.
- Add generated-app UI helper models to `views.py` while preserving legacy model exports for heavier production modules.

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
