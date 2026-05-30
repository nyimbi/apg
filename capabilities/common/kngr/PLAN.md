# KNGR Capability Plan

## Objectives

- Make KNGR a coherent APG capability packet with specification, usage docs,
  executable contract, runtime surfaces, UI models, package evidence, and
  focused verification.
- Keep the generated Python target dependency-light while exposing clear
  adapter boundaries for graph storage, NLP, ontology, search, metadata,
  governance, audit, metrics, cache, auth, AI-core, and Bytewax.
- Preserve the existing service API shape where practical so generated
  applications can keep using KNGR while the contract becomes more complete.

## Build Plan

1. Document the capability:
   - Add `README.md` with purpose, runtime surfaces, lifecycle, example,
     guardrails, and composition notes.
   - Add `SPECIFICATION.md` with functional scope, configuration, rule engine,
     UI contract, adapter boundaries, and non-goals.
   - Add this `PLAN.md` as the implementation plan and verification record.
   - Keep `cap_spec.md` aligned as the short APG capability summary.

2. Expand the executable contract:
   - Add lifecycle configuration sections for sources, entities,
     relationships, enrichment, reasoning, curation, publication, security,
     governance, observability, adapters, UI, and theme.
   - Add deterministic guardrails across every lifecycle stage.
   - Add route metadata for all expected generated-app screens.
   - Add theme components for every screen family.

3. Wire runtime surfaces:
   - Apply the expanded rules inside `KngrService`.
   - Keep source/entity/relationship/enrichment/reasoning/curation/publication
     workflows dependency-light and tenant-scoped.
   - Add aggregate graph listing and richer UI view models.
   - Expose review fields through API helpers.

4. Refresh package evidence:
   - Make `app.py` derive semantic metadata from the current contract.
   - Regenerate `semantic_model.json`, `package_manifest.json`, and
     `release_report.json`.

5. Verify and review:
   - Run targeted `py_compile` on KNGR modules and tests.
   - Run focused KNGR pytest files only.
   - Run APG implementation audit and publish-plan for KNGR.
   - Search for stale package markers and overclaim language in KNGR files.
   - Run `git diff --check` for KNGR and the progress log.

## Battery-Conscious Verification

This slice intentionally avoids the full repository pytest suite and live
adapter execution. The verification target is a coherent, importable,
package-backed KNGR capability with executable lifecycle guardrails and current
package evidence.
