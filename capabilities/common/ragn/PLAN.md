# RAGN Capability Plan

## Objectives

- Make RAGN a coherent APG capability packet with specification, usage docs,
  executable contract, generated-app runtime, API helpers, UI models, package
  evidence, and focused verification.
- Extend RAGN to the current APG guardrail standard: first-class AI-agent
  composition, provider-neutral runtime manifests, Bytewax lifecycle batches,
  deterministic governance rules, route metadata, theme components, and
  executable state.
- Keep generated-app imports dependency-light while preserving `service.py` as
  the heavier production adapter surface.
- Replace stale static package evidence with metadata derived from the current
  contract.

## Build Plan

1. Document the capability:
   - Replace root `README.md` with current generated-app behavior.
   - Add `SPECIFICATION.md` for functional scope and adapter boundaries.
   - Add this `PLAN.md` as the implementation and verification plan.
   - Replace `cap_spec.md` with the current packet summary.

2. Expand the executable contract:
   - Add lifecycle configuration sections for knowledge bases, documents,
     chunking, retrieval, generation, conversations, citations, curation,
     security, agents, streaming, governance, observability, adapters, UI, and
     theme.
   - Add deterministic guardrails across each lifecycle stage.
   - Add first-class RAG-agent guardrails for supported runtime, role, scope,
     owner, purpose, contribution disclosure, and privileged-role approval.
   - Add Bytewax lifecycle stream metadata for RAG lifecycle batches.
   - Add 14 generated-app UI routes and component theme definitions.

3. Wire generated-app runtime surfaces:
   - Add `rag_runtime.py` with dependency-light lifecycle records and service
     workflows.
   - Add provider-neutral RAG-agent records, Bytewax lifecycle batch records,
     registration, validation, listing, dashboard summaries, and audit events.
   - Point `api.py` at the generated-app runtime instead of importing the
     heavier production service stack.
   - Replace `views.py` with dependency-light view models, including agent
     roster and lifecycle batch monitor models.
   - Replace `__init__.py` registration metadata with current contract-derived
     composition data.

4. Refresh package evidence:
   - Make `app.py` derive semantic metadata from the current contract.
   - Regenerate `semantic_model.json`, `package_manifest.json`, and
     `release_report.json`.

5. Verify and review:
   - Run targeted `py_compile` on RAGN generated-app modules and tests.
   - Run focused RAGN package and lifecycle tests only.
   - Run APG implementation audit and publish-plan for RAGN.
   - Search primary RAGN package files for stale package markers and overclaim
     language.
   - Run `git diff --check` for RAGN and the progress log.

## Battery-Conscious Verification

This slice intentionally avoids the full repository pytest suite, live model
inference, vector indexes, async database setup, and browser UI checks. The
verification target is a coherent, importable, package-backed RAGN capability
with executable lifecycle guardrails and current package evidence.
